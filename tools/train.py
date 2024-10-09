import torch
from tqdm.auto import tqdm
from sklearn.metrics import classification_report

from .test import do_test


def do_train(cfg, model, dataloader_train, dataloader_test, optimizer, scheduler, criterion, logger):
    scaler = torch.cuda.amp.GradScaler()
    iters = len(dataloader_train)
    bar = tqdm(range(cfg.SOLVER.MAX_EPOCHS))
    best_score = 0
    for epoch in bar:
        running_loss = 0.0
        num_sample = 0
        y_true = []
        y_pred = []
        optimizer.zero_grad()
        model.train()
        for i, (inputs, labels) in tqdm(enumerate(dataloader_train), total=iters):
            y_true.append(labels[labels!=-100])

            # move data to GPU
            for key in inputs.keys():
                inputs[key] = inputs[key].to(cfg.MODEL.DEVICE)
            labels = labels.to(cfg.MODEL.DEVICE)
            
            # Forward
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(**inputs)
                loss = criterion(logits.permute(0,2,1), labels)
                loss = loss / cfg.SOLVER.ITERS_TO_ACCUMULATE
            
            # backward
            scaler.scale(loss).backward()
            if (i + 1) % cfg.SOLVER.ITERS_TO_ACCUMULATE == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # statistics
            num_sample += int((labels != -100).sum())
            running_loss += loss.item() * int((labels != -100).sum()) * cfg.SOLVER.ITERS_TO_ACCUMULATE
            y_pred.append(torch.argmax(logits, dim=-1)[labels!=-100].detach().cpu())
            bar.set_description(f"running_loss:{(running_loss / num_sample):.4f}")

        # scheduler.step()

        y_true, y_pred = torch.cat(y_true).numpy().astype('int')+1, torch.cat(y_pred).numpy().astype('int')+1
        loss_train = running_loss / num_sample
        performance_train = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
        performance_train_str = 'Train performance:\n' + classification_report(y_true, y_pred, zero_division=0, digits=3)
        loss_test, performance_test, performance_test_str = do_test(cfg, model, dataloader_test, criterion, epoch)

        logger.info(f"Epoch: {epoch+1}")
        logger.info(f"Train loss:{loss_train:.4f}")
        logger.info(f"Test loss:{loss_test:.4f}")
        logger.info(performance_train_str)
        logger.info(performance_test_str)

        # Save Model
        if performance_test['accuracy'] >= best_score:
            best_score = performance_test['accuracy']
            # Save Model
            logger.info(f"Saving model checpoint epoch {epoch+1}")
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                f"{cfg.OUTPUT_DIR}/checkpoint.pt",
            )