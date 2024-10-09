import torch
from sklearn.metrics import classification_report

def do_test(cfg, model, dataloader, criterion, epoch):
    running_loss = 0.0
    num_sample = 0
    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            y_true.append(labels[labels!=-100])

            # move data to GPU
            for key in inputs.keys():
                inputs[key] = inputs[key].to(cfg.MODEL.DEVICE)
            labels = labels.to(cfg.MODEL.DEVICE)

            # Forward
            logits = model(**inputs)
            loss = criterion(logits.permute(0,2,1), labels)

            # statistics
            num_sample += int((labels != -100).sum())
            running_loss += loss.item() * int((labels != -100).sum())
            y_pred.append(torch.argmax(logits, dim=-1)[labels!=-100].detach().cpu())

    y_true, y_pred = torch.cat(y_true).numpy().astype('int')+1, torch.cat(y_pred).numpy().astype('int')+1

    loss = running_loss / num_sample
    performance = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    performance_str = 'Test performance:\n' + classification_report(y_true, y_pred, zero_division=0, digits=3)

    return loss, performance, performance_str