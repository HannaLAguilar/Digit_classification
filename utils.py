import time
import numpy as np
import torch


def num_class(dataset):
    classes_name = dataset.class_to_idx.keys()
    classes_idx = dataset.class_to_idx.values()
    classes_count = []
    for ii in classes_idx:
        a = len(np.where(np.array(dataset.targets) == ii)[0])
        classes_count.append(a)
    return dict(zip(classes_name, classes_count))


def train_valid_test(dataset, train_ratio, valid_ratio):
    train_ratio = train_ratio - valid_ratio
    dataset_num = len(dataset)
    train_num = int(train_ratio * dataset_num)
    valid_num = int(valid_ratio * dataset_num)
    indices = list(range(dataset_num))
    np.random.shuffle(indices)
    return indices[:train_num], indices[train_num: train_num + valid_num], indices[train_num + valid_num:]


def train_model(epochs, model, data_loaders, criterion, optimizer, device):
    since = time.time()
    best_acc = 0

    train_loss_history, valid_loss_history = [], []
    valid_accuracy_history = []
    for epoch in range(epochs):
        train_loss, valid_loss = 0, 0
        print('Epoch: {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        # Train
        model.train()
        for images, labels in data_loaders['train']:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        # Valid
        model.eval()
        valid_corrects = 0
        for images, labels in data_loaders['valid']:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            valid_loss += loss.item() * images.size(0)

            _, preds = torch.max(output, 1)
            valid_corrects += (preds == labels).sum().item()
        accuracy = valid_corrects / len(data_loaders['valid'].sampler)

        # calculate average losses
        train_loss = train_loss / len(data_loaders['train'].sampler)
        valid_loss = valid_loss / len(data_loaders['valid'].sampler)

        # history average losses
        train_loss_history.append(train_loss)
        valid_loss_history.append(valid_loss)

        # valid accuracy history
        valid_accuracy_history.append(accuracy)

        print('Train loss:{:.3}  Valid loss:{:.3}  Valid acc:{:.1%}'.format(train_loss, valid_loss, accuracy))
        print()

        # save model
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), 'classifier_digit.pt')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Valid Acc: {:.1%}'.format(best_acc))
    return train_loss_history, valid_loss_history, valid_accuracy_history


def test_model(classes, model, data_loaders, device):
    print('\nTEST')
    num = len(classes)
    class_correct = list(range(num))
    class_total = list(range(num))

    model.eval()
    for images, labels in data_loaders['test']:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _, preds = torch.max(output, 1)
        corrects = preds == labels

        # calculate test accuracy for each object class
        for ii in range(len(labels.data)):
            label = labels.data[ii]
            class_correct[label] += corrects[ii].item()
            class_total[label] += 1
    accuracy = np.sum(class_correct)/np.sum(class_total)
    class_acc = [class_correct[ii] / class_total[ii] for ii in range(num)]

    for ii in range(num):
        print('Test accuracy of {}: {:.1%}'.format(classes[ii], class_acc[ii]))
    print('\nGlobal accuracy: {:.1%}'.format(accuracy))
    return accuracy, class_acc
