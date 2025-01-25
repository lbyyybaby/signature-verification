import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import random
import os
import numpy as np

# Set seed for reproducibility
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def tune_threshold(model, val_loader, device):
    # Precompute distances and labels
    model.eval()
    all_distances = []
    all_labels = []

    with torch.no_grad():
        for img1, img2, labels in val_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            output1, output2 = model(img1, img2)
            all_distances.extend(F.pairwise_distance(output1, output2).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_distances = np.array(all_distances)
    all_labels = np.array(all_labels)

    # Determine threshold range dynamically
    min_dist, max_dist = np.min(all_distances), np.max(all_distances)
    thresholds = np.linspace(min_dist, max_dist, 100)

    # Tune threshold
    best_score = 0
    best_threshold = 0

    for threshold in thresholds:
        predictions = (all_distances < threshold).astype(float)
        val_f1 = f1_score(all_labels, predictions)

        if val_f1 > best_score:
            best_score = val_f1
            best_threshold = threshold

    return best_threshold

# Evaluate the model
def evaluate(model, val_loader, criterion, device, threshold):
    model.eval()
    val_loss = 0.0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for img1, img2, labels in val_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, labels)
            val_loss += loss.item()

            euclidean_distance = F.pairwise_distance(output1, output2)
            predictions = (euclidean_distance < threshold).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(all_labels, all_predictions)
    val_precision = precision_score(all_labels, all_predictions)
    val_recall = recall_score(all_labels, all_predictions)
    val_f1 = f1_score(all_labels, all_predictions)
    
    return avg_val_loss, val_accuracy, val_precision, val_recall, val_f1


# Train and evaluate the model
def train_and_evaluate(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, device, save_path):
    all_epoch_scores = []
    best_epoch_stats = {}
    best_val_f1 = 0    

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for img1, img2, labels in tqdm(train_loader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        best_threshold = tune_threshold(model, val_loader, device)
        avg_val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(model, val_loader, criterion, device, threshold=best_threshold)
        scheduler.step()
        
        epoch_scores = {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "accuracy": val_accuracy,
                    "precision": val_precision,
                    "recall": val_recall,
                    "f1": val_f1,
                }
        all_epoch_scores.append(epoch_scores)
        print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | Tuned Thresold: {best_threshold:.4f} | Val Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
        
        if best_val_f1 < val_f1:
            best_val_f1 = val_f1
            best_epoch_stats = {
                "epoch": epoch + 1,
                "accuracy": val_accuracy,
                "precision": val_precision,
                "recall": val_recall,
                "f1": val_f1,
            }
            torch.save(model.state_dict(), save_path)
            print(f'Saved the best model with F1: {best_val_f1:.4f}')
            
    return best_threshold, all_epoch_scores, best_epoch_stats

# Test the model
def test(model, test_loader, criterion, device, threshold):
    model.eval()
    test_loss = 0.0
    all_labels = []
    all_predictions = []
    test_stats = {}
    
    with torch.no_grad():
        for img1, img2, labels in test_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, labels)
            test_loss += loss.item()

            euclidean_distance = F.pairwise_distance(output1, output2)
            predictions = (euclidean_distance < threshold).float()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = accuracy_score(all_labels, all_predictions)
    test_precision = precision_score(all_labels, all_predictions)
    test_recall = recall_score(all_labels, all_predictions)
    test_f1 = f1_score(all_labels, all_predictions)
    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')
    
    cm = confusion_matrix(all_labels, all_predictions)
    test_stats = {
                "accuracy": test_accuracy,
                "precision": test_precision,
                "recall": test_recall,
                "f1": test_f1,
                }
    
    return test_stats, cm