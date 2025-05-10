import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import resnet50
from torchvision.ops import box_iou
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import requests
import tarfile
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random
import os.path as osp
from pathlib import Path

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# VOC Labels
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
# Number of classes (excluding background)
NUM_CLASSES = len(VOC_CLASSES) - 1
CLASS_TO_IDX = {}
idx = 0
for i, cls_name in enumerate(VOC_CLASSES):
    if cls_name != 'background':
        CLASS_TO_IDX[cls_name] = idx
        idx += 1

print(f"Class to index mapping: {CLASS_TO_IDX}")
print(f"Number of classes (excluding background): {len(CLASS_TO_IDX)}")

# YOLO grid parameters
GRID_SIZE = 13  # Output grid size
NUM_BBOXES = 3   # Number of bounding boxes per grid cell
NUM_ATTRIB = 5   # x, y, w, h, objectness

print(f"Total output dimension per grid cell: {NUM_BBOXES * (NUM_ATTRIB + NUM_CLASSES)}")
print(f"Number of classes: {NUM_CLASSES}")
print(f"Total feature map dimensions: {GRID_SIZE}x{GRID_SIZE}x{NUM_BBOXES}x{NUM_ATTRIB + NUM_CLASSES}")

# Download and prepare Pascal VOC dataset
def download_pascal_voc():
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)

    voc_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    tar_file = os.path.join(data_dir, "VOCtrainval_11-May-2012.tar")
    extracted_dir = os.path.join(data_dir, "VOCdevkit")

    if os.path.exists(extracted_dir):
        print("Dataset already downloaded and extracted.")
        return os.path.join(extracted_dir, "VOC2012")

    # Download the dataset
    print("Downloading Pascal VOC dataset...")
    response = requests.get(voc_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024

    with open(tar_file, 'wb') as f:
        for data in tqdm(response.iter_content(block_size), total=total_size//block_size, desc="Downloading"):
            f.write(data)

    # Extract the dataset
    print("Extracting dataset...")
    with tarfile.open(tar_file) as tar:
        tar.extractall(data_dir)

    print("Dataset downloaded and extracted successfully.")
    return os.path.join(extracted_dir, "VOC2012")

# Custom VOC Dataset
class PascalVOCDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, subset_size=444):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split

        # Get image sets
        split_file = os.path.join(root_dir, 'ImageSets', 'Main', f'{split}.txt')
        with open(split_file, 'r') as f:
            image_ids = [line.strip() for line in f.readlines()]

        # Limit to subset size if specified
        if subset_size > 0:
            self.image_ids = image_ids[:subset_size]
        else:
            self.image_ids = image_ids

        print(f"Loaded {len(self.image_ids)} {split} images")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Load image
        img_path = os.path.join(self.root_dir, 'JPEGImages', f'{image_id}.jpg')
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size

        # Load annotation
        anno_path = os.path.join(self.root_dir, 'Annotations', f'{image_id}.xml')
        tree = ET.parse(anno_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in CLASS_TO_IDX:
                continue

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Normalize coordinates
            xmin /= orig_width
            ymin /= orig_height
            xmax /= orig_width
            ymax /= orig_height

            # YOLO format: center_x, center_y, width, height (normalized)
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            boxes.append([center_x, center_y, width, height])
            labels.append(CLASS_TO_IDX[class_name])

        if len(boxes) == 0:
            boxes = torch.zeros((1, 4))
            labels = torch.zeros(1, dtype=torch.long)
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': image_id,
                'orig_size': (orig_height, orig_width)
            }

            if self.transform:
                image = self.transform(image)

            return image, target

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'orig_size': (orig_height, orig_width)
        }

        if self.transform:
            image = self.transform(image)

        # Convert target to YOLO format target grid
        yolo_target = self.encode_target(target)
        return image, yolo_target

    def encode_target(self, target):
        """Convert detection target to YOLO format."""
        boxes = target['boxes']  # [N, 4] in normalized coordinates (cx, cy, w, h)
        labels = target['labels']  # [N]

        # Initialize target tensor
        # [grid_size, grid_size, num_bboxes, (5 + num_classes)]
        encoded = torch.zeros((GRID_SIZE, GRID_SIZE, NUM_BBOXES, 5 + NUM_CLASSES))

        for i in range(len(boxes)):
            # Extract box coordinates
            cx, cy, w, h = boxes[i]
            label = labels[i]

            # Find which grid cell this box belongs to
            grid_x = int(cx * GRID_SIZE)
            grid_y = int(cy * GRID_SIZE)

            # Constrain to valid grid indices
            grid_x = min(grid_x, GRID_SIZE-1)
            grid_y = min(grid_y, GRID_SIZE-1)

            # Convert cx, cy to be relative to grid cell
            cell_cx = cx * GRID_SIZE - grid_x
            cell_cy = cy * GRID_SIZE - grid_y

            # Fill target for each of the anchor boxes
            for b in range(NUM_BBOXES):
                # Box coordinates and objectness
                encoded[grid_y, grid_x, b, 0] = cell_cx
                encoded[grid_y, grid_x, b, 1] = cell_cy
                encoded[grid_y, grid_x, b, 2] = w
                encoded[grid_y, grid_x, b, 3] = h
                encoded[grid_y, grid_x, b, 4] = 1.0  # objectness score

                # Class one-hot encoding - ensure index is within bounds
                class_idx = min(label, NUM_CLASSES-1)  # Prevent out of bounds indexing
                encoded[grid_y, grid_x, b, 5 + class_idx] = 1.0

        return encoded

# ResNet-50 + YOLO Detection Model
class ResNet50YOLO(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(ResNet50YOLO, self).__init__()
        # Load ResNet-50 backbone
        resnet = resnet50(pretrained=pretrained)

        # Remove the final FC layer and average pooling
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze the first few layers of backbone
        for i, param in enumerate(self.backbone.parameters()):
            if i < 100:  # Freeze initial layers
                param.requires_grad = False

        # Detection head
        self.detection_head = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),

            nn.Conv2d(512, NUM_BBOXES * (5 + num_classes), kernel_size=1)
        )

    def forward(self, x):
        # Input shape: [batch_size, 3, H, W]
        features = self.backbone(x)  # [batch_size, 2048, H/32, W/32]

        # Run detection head
        detection = self.detection_head(features)  # [batch_size, NUM_BBOXES * (5 + num_classes), H/32, W/32]

        # Reshape output for loss computation
        batch_size = x.size(0)
        detection = detection.permute(0, 2, 3, 1).contiguous()
        detection = detection.view(batch_size, GRID_SIZE, GRID_SIZE, NUM_BBOXES, 5 + NUM_CLASSES)

        return detection

# YOLO Loss Function
class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, predictions, targets):
        
        batch_size = predictions.size(0)

        # Apply sigmoid to the objectness score and class probabilities
        pred_xy = torch.sigmoid(predictions[..., 0:2])
        pred_wh = predictions[..., 2:4]
        pred_obj = predictions[..., 4:5]
        pred_cls = predictions[..., 5:]

        # Get target components
        targ_xy = targets[..., 0:2]
        targ_wh = targets[..., 2:4]
        targ_obj = targets[..., 4:5]
        targ_cls = targets[..., 5:]

        # Expand the mask to match dimensions for operations
        obj_mask = targ_obj > 0
        noobj_mask = ~obj_mask

        # Count responsible cells for normalization
        num_responsible_cells = max(1, torch.sum(obj_mask).item())

        # Reshape the mask to match the dimensions of the coordinates
        obj_mask_xy = obj_mask.expand_as(targ_xy)

        # Use MSE loss for x, y coordinates
        loss_xy = self.mse_loss(
            pred_xy[obj_mask_xy],
            targ_xy[obj_mask_xy]
        ) / num_responsible_cells

        # Create mask for width and height
        obj_mask_wh = obj_mask.expand_as(targ_wh)

        # Use MSE loss for width and height (square root to reduce impact of large boxes)
        loss_wh = self.mse_loss(
            torch.sqrt(torch.clamp(pred_wh[obj_mask_wh], min=1e-6)),
            torch.sqrt(torch.clamp(targ_wh[obj_mask_wh], min=1e-6))
        ) / num_responsible_cells

        # Objectness loss
        loss_obj = self.mse_loss(
            pred_obj[obj_mask],
            targ_obj[obj_mask]
        ) / num_responsible_cells

        # No object loss (penalize predictions where there should be no objects)
        loss_noobj = self.mse_loss(
            pred_obj[noobj_mask],
            targ_obj[noobj_mask]
        ) / max(1, torch.sum(noobj_mask).item())

        # Classification loss (only for cells that contain objects)
        # Create mask for class predictions
        obj_mask_cls = obj_mask.expand_as(targ_cls)

        loss_cls = nn.functional.binary_cross_entropy_with_logits(
            pred_cls[obj_mask_cls],
            targ_cls[obj_mask_cls],
            reduction='sum'
        ) / num_responsible_cells

        # Combine all losses with weighting
        total_loss = (
            self.lambda_coord * (loss_xy + loss_wh) +
            loss_obj +
            self.lambda_noobj * loss_noobj +
            loss_cls
        )

        # For monitoring individual loss components
        loss_components = {
            'loss': total_loss.item(),
            'loss_xy': loss_xy.item(),
            'loss_wh': loss_wh.item(),
            'loss_obj': loss_obj.item(),
            'loss_noobj': loss_noobj.item(),
            'loss_cls': loss_cls.item()
        }

        return total_loss, loss_components

# Calculation of Precision-Recall metrics
def calculate_precision_recall(all_pred_boxes, all_pred_scores, all_pred_classes, 
                               all_gt_boxes, all_gt_classes, iou_thresholds=[0.5], 
                               confidence_threshold=0.5):
   
    results = {iou: {c: {'TP': 0, 'FP': 0, 'FN': 0} for c in range(NUM_CLASSES)} for iou in iou_thresholds}
    
    # Process each image
    for i in range(len(all_pred_boxes)):
        pred_boxes = all_pred_boxes[i]
        pred_scores = all_pred_scores[i]
        pred_classes = all_pred_classes[i]
        gt_boxes = all_gt_boxes[i]
        gt_classes = all_gt_classes[i]
        
        # Filter predictions by confidence threshold
        if len(pred_boxes) > 0:
            confident_mask = pred_scores >= confidence_threshold
            pred_boxes = pred_boxes[confident_mask]
            pred_scores = pred_scores[confident_mask]
            pred_classes = pred_classes[confident_mask]
        
        # For each IoU threshold
        for iou_threshold in iou_thresholds:
            # Group predictions by class
            pred_by_class = {}
            for j in range(len(pred_boxes)):
                cls = int(pred_classes[j])
                if cls not in pred_by_class:
                    pred_by_class[cls] = []
                pred_by_class[cls].append((pred_boxes[j], pred_scores[j]))
            
            # Group ground truths by class
            gt_by_class = {}
            gt_matched = {}  # Keep track of matched ground truths
            
            for j in range(len(gt_boxes)):
                cls = int(gt_classes[j])
                if cls not in gt_by_class:
                    gt_by_class[cls] = []
                box_idx = len(gt_by_class[cls])
                gt_by_class[cls].append(gt_boxes[j])
                gt_matched[(cls, box_idx)] = False
            
            # For each class
            for cls in range(NUM_CLASSES):
                # Count false negatives (unmatched ground truths)
                class_gt_boxes = gt_by_class.get(cls, [])
                results[iou_threshold][cls]['FN'] += len(class_gt_boxes)
                
                # Process predictions for this class
                class_pred_items = pred_by_class.get(cls, [])
                
                if not class_pred_items:
                    continue
                    
                # Sort predictions by confidence
                class_pred_items.sort(key=lambda x: x[1], reverse=True)
                
                for pred_box, _ in class_pred_items:
                    # If no ground truth for this class, all predictions are false positives
                    if cls not in gt_by_class or len(gt_by_class[cls]) == 0:
                        results[iou_threshold][cls]['FP'] += 1
                        continue
                    
                    # Calculate IoUs with all ground truths of this class
                    ious = box_iou_numpy(np.array([pred_box]), np.array(gt_by_class[cls]))[0]
                    
                    if len(ious) == 0:
                        results[iou_threshold][cls]['FP'] += 1
                        continue
                    
                    # Find best matching ground truth
                    best_iou = np.max(ious)
                    best_idx = np.argmax(ious)
                    
                    # Check if IoU exceeds threshold and ground truth not already matched
                    if best_iou >= iou_threshold and not gt_matched.get((cls, best_idx), True):
                        results[iou_threshold][cls]['TP'] += 1
                        results[iou_threshold][cls]['FN'] -= 1  # Decrement FN since we matched this GT
                        gt_matched[(cls, best_idx)] = True
                    else:
                        results[iou_threshold][cls]['FP'] += 1
    
    # Calculate precision and recall for each class and IoU threshold
    metrics = {}
    for iou in iou_thresholds:
        metrics[iou] = {}
        for cls in range(NUM_CLASSES):
            tp = results[iou][cls]['TP']
            fp = results[iou][cls]['FP']
            fn = results[iou][cls]['FN']
            
            # Calculate precision and recall
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            cls_name = VOC_CLASSES[cls+1] if cls+1 < len(VOC_CLASSES) else f"class_{cls}"
            metrics[iou][cls_name] = {
                'precision': precision,
                'recall': recall,
                'TP': tp,
                'FP': fp,
                'FN': fn
            }
    
    return metrics

# Function to calculate PR curve points for plotting
def calculate_pr_curve(all_pred_boxes, all_pred_scores, all_pred_classes, 
                       all_gt_boxes, all_gt_classes, 
                       iou_threshold=0.5, class_id=None):
   
    # Collect all predictions and ground truths
    all_predictions = []
    
    # Count total ground truths per class
    gt_counts = {c: 0 for c in range(NUM_CLASSES)}
    
    # Process ground truths
    for i in range(len(all_gt_boxes)):
        gt_boxes = all_gt_boxes[i]
        gt_classes = all_gt_classes[i]
        
        for j in range(len(gt_classes)):
            cls = int(gt_classes[j])
            gt_counts[cls] = gt_counts.get(cls, 0) + 1
    
    # Process predictions
    for i in range(len(all_pred_boxes)):
        pred_boxes = all_pred_boxes[i]
        pred_scores = all_pred_scores[i]
        pred_classes = all_pred_classes[i]
        gt_boxes = all_gt_boxes[i]
        gt_classes = all_gt_classes[i]
        
        if len(pred_boxes) == 0:
            continue
            
        # For each prediction in this image
        for j in range(len(pred_classes)):
            cls = int(pred_classes[j])
            
            if class_id is not None and cls != class_id:
                continue
                
            score = pred_scores[j]
            pred_box = pred_boxes[j]
            
            # Find matching ground truths for this class
            matching_gt_indices = [k for k in range(len(gt_classes)) if gt_classes[k] == cls]
            
            # If no matching ground truths, it's a false positive
            if not matching_gt_indices:
                all_predictions.append((cls, score, 0))  # FP
                continue
                
            # Calculate IoUs with matching ground truths
            matching_gt_boxes = [gt_boxes[k] for k in matching_gt_indices]
            ious = box_iou_numpy(np.array([pred_box]), np.array(matching_gt_boxes))[0]
            
            # If any IoU exceeds threshold, it's a true positive
            if np.max(ious) >= iou_threshold:
                all_predictions.append((cls, score, 1))  # TP
            else:
                all_predictions.append((cls, score, 0))  # FP
    
    if not all_predictions or (class_id is not None and all(p[0] != class_id for p in all_predictions)):
        # Return empty arrays
        return np.array([]), np.array([]), "unknown"
    
    # If class_id specified, filter predictions
    if class_id is not None:
        filtered_predictions = [(cls, score, correct) for cls, score, correct in all_predictions if cls == class_id]
        class_name = VOC_CLASSES[class_id+1] if class_id+1 < len(VOC_CLASSES) else f"class_{class_id}"
        total_gt = gt_counts.get(class_id, 0)
    else:
        filtered_predictions = all_predictions
        class_name = "all classes"
        total_gt = sum(gt_counts.values())
    
    # If no predictions after filtering
    if not filtered_predictions:
        return np.array([]), np.array([]), class_name
    
    # Sort by confidence score in descending order
    filtered_predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate cumulative TP and FP
    tp_cumsum = np.cumsum([p[2] for p in filtered_predictions])
    fp_cumsum = np.cumsum([1 - p[2] for p in filtered_predictions])
    
    # Calculate precision and recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / total_gt if total_gt > 0 else np.zeros_like(tp_cumsum)
    
    # Add point at recall=0, precision=1
    precision = np.concatenate(([1], precision))
    recall = np.concatenate(([0], recall))
    
    return precision, recall, class_name

# Function to visualize PR curve
def plot_pr_curve(all_pred_boxes, all_pred_scores, all_pred_classes, 
                 all_gt_boxes, all_gt_classes, 
                 iou_threshold=0.5, class_ids=None):
   
    plt.figure(figsize=(10, 8))
    
    if class_ids is None:
        # Plot average PR curve across all classes
        precision, recall, class_name = calculate_pr_curve(
            all_pred_boxes, all_pred_scores, all_pred_classes,
            all_gt_boxes, all_gt_classes, iou_threshold
        )
        plt.plot(recall, precision, label=f"{class_name}")
    else:
        # Plot PR curves for specified classes
        for class_id in class_ids:
            precision, recall, class_name = calculate_pr_curve(
                all_pred_boxes, all_pred_scores, all_pred_classes,
                all_gt_boxes, all_gt_classes, iou_threshold, class_id
            )
            
            if len(precision) > 0:
                plt.plot(recall, precision, label=f"{class_name}")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (IoU={iou_threshold})')
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.savefig('precision_recall_curve.png')
    plt.show()

# Function to save and download model
def save_and_download_model(model, model_save_path="yolo_model_final.pth"):
  
    # Save model weights
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    try:
        from google.colab import files
        files.download(model_save_path)
        print(f"Download initiated for {model_save_path}")
    except:
        print(f"Model saved at {model_save_path}")
        
    return model_save_path

# Function to evaluate precision and recall metrics
def evaluate_precision_recall(model, val_loader):
    
    model.eval()
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_classes = []
    all_gt_boxes = []
    all_gt_classes = []
    
    print("Collecting predictions and ground truths for PR evaluation...")
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Evaluating"):
            # Move data to device
            images = images.to(device)
            
            # Forward pass
            predictions = model(images)
            
            # Decode predictions
            pred_boxes, pred_scores, pred_classes = decode_predictions(predictions)
            
            # Process each image in batch
            batch_size = images.size(0)
            for i in range(batch_size):
                # Get prediction for this image
                boxes_i = pred_boxes[i]
                scores_i = pred_scores[i]
                classes_i = pred_classes[i]
                
                # Apply NMS
                if len(boxes_i) > 0:
                    boxes_i = boxes_i.cpu().numpy()
                    scores_i = scores_i.cpu().numpy()
                    classes_i = classes_i.cpu().numpy()
                    
                    try:
                        boxes_i, scores_i, classes_i = apply_nms(boxes_i, scores_i, classes_i)
                    except Exception as e:
                        print(f"NMS error: {e}")
                        # If NMS fails, keep original predictions
                        pass
                
                # Get ground truth for this image
                target = targets[i]
                
                # Extract ground truth boxes and classes
                gt_boxes_list = []
                gt_classes_list = []
                
                for gy in range(GRID_SIZE):
                    for gx in range(GRID_SIZE):
                        for b in range(NUM_BBOXES):
                            if target[gy, gx, b, 4] > 0:
                                # Extract box coordinates in YOLO format
                                tx = target[gy, gx, b, 0]
                                ty = target[gy, gx, b, 1]
                                tw = target[gy, gx, b, 2]
                                th = target[gy, gx, b, 3]
                                
                                bx = (gx + tx) / GRID_SIZE
                                by = (gy + ty) / GRID_SIZE
                                bw = tw
                                bh = th
                                
                                x1 = max(0.0, bx - bw/2)
                                y1 = max(0.0, by - bh/2)
                                x2 = min(1.0, bx + bw/2)
                                y2 = min(1.0, by + bh/2)
                                
                                class_probs = target[gy, gx, b, 5:]
                                class_id = torch.argmax(class_probs).item()
                                
                                gt_boxes_list.append([x1, y1, x2, y2])
                                gt_classes_list.append(class_id)
                
                # Add to evaluation lists
                all_pred_boxes.append(boxes_i)
                all_pred_scores.append(scores_i)
                all_pred_classes.append(classes_i)
                all_gt_boxes.append(np.array(gt_boxes_list))
                all_gt_classes.append(np.array(gt_classes_list))
    
    # Calculate precision and recall metrics
    metrics = calculate_precision_recall(
        all_pred_boxes, all_pred_scores, all_pred_classes, 
        all_gt_boxes, all_gt_classes
    )
    
    # Print results
    print("\nPrecision-Recall Metrics (IoU=0.5):")
    print("-" * 60)
    print(f"{'Class':20s} | {'Precision':10s} | {'Recall':10s}")
    print("-" * 60)
    
    mean_precision = 0
    mean_recall = 0
    count = 0
    
    for cls_name, values in metrics[0.5].items():
        print(f"{cls_name:20s} | {values['precision']:10.4f} | {values['recall']:10.4f}")
        mean_precision += values['precision']
        mean_recall += values['recall']
        count += 1
    
    print("-" * 60)
    print(f"{'Mean':20s} | {mean_precision/count:10.4f} | {mean_recall/count:10.4f}")
    
    # Plot PR curves for some classes
    top_classes = sorted(
        [(cls_name, values['precision'] + values['recall']) 
         for cls_name, values in metrics[0.5].items()],
        key=lambda x: x[1], 
        reverse=True
    )[:5]
    
    top_class_indices = [list(metrics[0.5].keys()).index(cls_name) for cls_name, _ in top_classes]
    
    # Plot PR curves
    plot_pr_curve(
        all_pred_boxes, all_pred_scores, all_pred_classes,
        all_gt_boxes, all_gt_classes,
        class_ids=top_class_indices
    )
    
    return all_pred_boxes, all_pred_scores, all_pred_classes, all_gt_boxes, all_gt_classes
def decode_predictions(predictions, conf_threshold=0.25):
    
    batch_size = predictions.size(0)
    device = predictions.device

    all_boxes = []
    all_scores = []
    all_class_ids = []

    for b in range(batch_size):
        boxes = []
        scores = []
        class_ids = []

        # Extract prediction components with sigmoid applied where needed
        pred = predictions[b]  # [grid_size, grid_size, num_bboxes, 5+num_classes]

        for cy in range(GRID_SIZE):
            for cx in range(GRID_SIZE):
                for b_idx in range(NUM_BBOXES):
                    tx = torch.sigmoid(pred[cy, cx, b_idx, 0])
                    ty = torch.sigmoid(pred[cy, cx, b_idx, 1])
                    tw = pred[cy, cx, b_idx, 2]
                    th = pred[cy, cx, b_idx, 3]
                    objectness = torch.sigmoid(pred[cy, cx, b_idx, 4])
                    class_probs = torch.sigmoid(pred[cy, cx, b_idx, 5:])

                    # Get best class
                    class_score, class_id = torch.max(class_probs, dim=0)
                    confidence = objectness * class_score

                    if confidence < conf_threshold:
                        continue

                    bx = (cx + tx) / GRID_SIZE
                    by = (cy + ty) / GRID_SIZE
                    bw = torch.exp(tw) / GRID_SIZE
                    bh = torch.exp(th) / GRID_SIZE

                    x1 = max(0.0, bx - bw/2)
                    y1 = max(0.0, by - bh/2)
                    x2 = min(1.0, bx + bw/2)
                    y2 = min(1.0, by + bh/2)

                    box_coords = []
                    for coord in [x1, y1, x2, y2]:
                        if isinstance(coord, torch.Tensor):
                            box_coords.append(coord.item())
                        else:
                            box_coords.append(coord)

                    boxes.append(box_coords)

                    if isinstance(confidence, torch.Tensor):
                        scores.append(confidence.item())
                    else:
                        scores.append(confidence)

                    if isinstance(class_id, torch.Tensor):
                        class_ids.append(class_id.item())
                    else:
                        class_ids.append(class_id)

        all_boxes.append(torch.tensor(boxes, device=device) if boxes else torch.zeros((0, 4), device=device))
        all_scores.append(torch.tensor(scores, device=device) if scores else torch.zeros(0, device=device))
        all_class_ids.append(torch.tensor(class_ids, device=device) if class_ids else torch.zeros(0, dtype=torch.int64, device=device))

    return all_boxes, all_scores, all_class_ids


# Apply Non-Maximum Suppression (NMS)
def apply_nms(boxes, scores, class_ids, iou_threshold=0.45):
    if len(boxes) == 0:
        return [], [], []

    # Convert to numpy arrays
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        class_ids = class_ids.detach().cpu().numpy()

    # Group by class
    indices_by_class = {}
    for i, class_id in enumerate(class_ids):
        if class_id not in indices_by_class:
            indices_by_class[class_id] = []
        indices_by_class[class_id].append(i)

    # Apply NMS for each class
    keep_boxes = []
    keep_scores = []
    keep_class_ids = []

    for class_id, indices in indices_by_class.items():
        cls_boxes = boxes[indices]
        cls_scores = scores[indices]

        # Sort by confidence
        sorted_idx = np.argsort(-cls_scores)
        keep_idx = []

        while len(sorted_idx) > 0:
            # Pick the box with highest confidence
            current_idx = sorted_idx[0]
            keep_idx.append(current_idx)

            if len(sorted_idx) == 1:
                break

            current_box = cls_boxes[current_idx:current_idx+1]
            rest_boxes = cls_boxes[sorted_idx[1:]]

            # Calculate IoU
            ious = box_iou_numpy(current_box, rest_boxes)

            mask = ious < iou_threshold

            sorted_idx = sorted_idx[1:][mask.flatten()]

        for idx in keep_idx:
            keep_boxes.append(cls_boxes[idx])
            keep_scores.append(cls_scores[idx])
            keep_class_ids.append(class_id)

    return np.array(keep_boxes), np.array(keep_scores), np.array(keep_class_ids)

# Helper function for numpy IoU calculation
def box_iou_numpy(box1, box2):
    # Calculate intersection areas
    x1 = np.maximum(box1[:, 0, None], box2[:, 0])
    y1 = np.maximum(box1[:, 1, None], box2[:, 1])
    x2 = np.minimum(box1[:, 2, None], box2[:, 2])
    y2 = np.minimum(box1[:, 3, None], box2[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calculate union areas
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    union = box1_area[:, None] + box2_area - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-6)
    return iou

# Evaluation functions
def calculate_mAP(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5):
    ap_per_class = {}

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0, ap_per_class

    # For each class
    for c in range(NUM_CLASSES):
        # Get predictions for this class
        mask = pred_classes == c
        class_pred_boxes = pred_boxes[mask] if len(pred_boxes) > 0 else np.array([])
        class_pred_scores = pred_scores[mask] if len(pred_scores) > 0 else np.array([])

        # Get ground truth for this class
        gt_mask = gt_classes == c
        class_gt_boxes = gt_boxes[gt_mask] if len(gt_boxes) > 0 else np.array([])

        if len(class_gt_boxes) == 0:
            continue

        if len(class_pred_boxes) == 0:
            ap_per_class[c] = 0.0
            continue

        sorted_indices = np.argsort(-class_pred_scores)
        class_pred_boxes = class_pred_boxes[sorted_indices]
        class_pred_scores = class_pred_scores[sorted_indices]

        # Create arrays for precision-recall curve
        tp = np.zeros(len(class_pred_boxes))
        fp = np.zeros(len(class_pred_boxes))

        # Create used flag for ground truth boxes
        gt_used = np.zeros(len(class_gt_boxes), dtype=bool)

        # For each prediction
        for i, pred_box in enumerate(class_pred_boxes):
            # If no ground truth left, it's a false positive
            if len(class_gt_boxes) == 0:
                fp[i] = 1
                continue

            # Calculate IoU with all ground truth boxes
            ious = box_iou_numpy(np.array([pred_box]), class_gt_boxes)[0]

            # Find max IoU and corresponding ground truth box
            max_iou = np.max(ious)
            max_idx = np.argmax(ious)

            # If IoU > threshold and ground truth not used, it's a true positive
            if max_iou >= iou_threshold and not gt_used[max_idx]:
                tp[i] = 1
                gt_used[max_idx] = True
            else:
                fp[i] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        # Calculate precision and recall
        precision = cum_tp / (cum_tp + cum_fp + 1e-6)
        recall = cum_tp / len(class_gt_boxes)

        # Calculate average precision
        ap = 0.0
        for r in np.arange(0, 1.1, 0.1):
            mask = recall >= r
            if np.any(mask):
                ap += np.max(precision[mask]) / 11

        ap_per_class[c] = ap

    # Calculate mAP
    if len(ap_per_class) > 0:
        mAP = sum(ap_per_class.values()) / len(ap_per_class)
    else:
        mAP = 0.0

    return mAP, ap_per_class

# Visualization function
def visualize_predictions(image, boxes, scores, class_ids, threshold=0.5):
    # Create a copy of the image
    image = image.copy()
    draw = ImageDraw.Draw(image)

    width, height = image.size

    # Draw each prediction
    for i in range(len(boxes)):
        if scores[i] < threshold:
            continue

        # Get box coordinates
        box = boxes[i]
        x1, y1, x2, y2 = box

        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)

        class_id = class_ids[i]
        class_name = VOC_CLASSES[class_id + 1]  # +1 because background is at index 0

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Draw label
        label = f"{class_name}: {scores[i]:.2f}"
        draw.text((x1, y1), label, fill="red")

    return image

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    best_map = 0.0
    history = {'train_loss': [], 'val_mAP': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_loss_components = {
            'loss_xy': 0.0,
            'loss_wh': 0.0,
            'loss_obj': 0.0,
            'loss_noobj': 0.0,
            'loss_cls': 0.0
        }

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for images, targets in train_bar:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)

            # Calculate loss
            loss, loss_components = criterion(predictions, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            for k, v in loss_components.items():
                if k in epoch_loss_components:
                    epoch_loss_components[k] += v

            train_bar.set_postfix({
                'loss': running_loss / (train_bar.n + 1)
            })

        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
        for k, v in epoch_loss_components.items():
            print(f"  {k}: {v / len(train_loader):.4f}")

        val_mAP = evaluate_model(model, val_loader)
        history['val_mAP'].append(val_mAP)

        print(f"Epoch {epoch+1}/{num_epochs}, Validation mAP: {val_mAP:.4f}")

        scheduler.step()

        # Save best model
        if val_mAP > best_map:
            best_map = val_mAP
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved best model with mAP: {best_map:.4f}")

    return history

# Evaluation function
def evaluate_model(model, val_loader):
    model.eval()
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_classes = []
    all_gt_boxes = []
    all_gt_classes = []

    val_bar = tqdm(val_loader, desc="Validating")

    with torch.no_grad():
        for images, targets in val_bar:
            images = images.to(device)

            predictions = model(images)

            # Decode predictions
            pred_boxes, pred_scores, pred_classes = decode_predictions(predictions)

            # Process each image in batch
            batch_size = images.size(0)
            for i in range(batch_size):
                boxes_i = pred_boxes[i]
                scores_i = pred_scores[i]
                classes_i = pred_classes[i]

                # Apply NMS
                if len(boxes_i) > 0:
                    boxes_i = boxes_i.cpu().numpy()
                    scores_i = scores_i.cpu().numpy()
                    classes_i = classes_i.cpu().numpy()

                    try:
                        boxes_i, scores_i, classes_i = apply_nms(boxes_i, scores_i, classes_i)
                    except Exception as e:
                        print(f"NMS error: {e}")
                        pass

                target = targets[i]

                gt_boxes_list = []
                gt_classes_list = []

                for gy in range(GRID_SIZE):
                    for gx in range(GRID_SIZE):
                        for b in range(NUM_BBOXES):
                            if target[gy, gx, b, 4] > 0:
                                tx = target[gy, gx, b, 0]
                                ty = target[gy, gx, b, 1]
                                tw = target[gy, gx, b, 2]
                                th = target[gy, gx, b, 3]

                                bx = (gx + tx) / GRID_SIZE
                                by = (gy + ty) / GRID_SIZE
                                bw = tw
                                bh = th

                                x1 = max(0.0, bx - bw/2)
                                y1 = max(0.0, by - bh/2)
                                x2 = min(1.0, bx + bw/2)
                                y2 = min(1.0, by + bh/2)

                                class_probs = target[gy, gx, b, 5:]
                                class_id = torch.argmax(class_probs).item()

                                gt_boxes_list.append([x1, y1, x2, y2])
                                gt_classes_list.append(class_id)

                # Add to evaluation lists
                all_pred_boxes.append(boxes_i)
                all_pred_scores.append(scores_i)
                all_pred_classes.append(classes_i)
                all_gt_boxes.append(np.array(gt_boxes_list))
                all_gt_classes.append(np.array(gt_classes_list))

    # Concatenate all predictions and ground truths
    try:
        pred_boxes = np.concatenate(all_pred_boxes) if all_pred_boxes and len(all_pred_boxes[0]) > 0 else np.array([])
        pred_scores = np.concatenate(all_pred_scores) if all_pred_scores and len(all_pred_scores[0]) > 0 else np.array([])
        pred_classes = np.concatenate(all_pred_classes) if all_pred_classes and len(all_pred_classes[0]) > 0 else np.array([])
    except ValueError as e:
        print(f"Warning in concatenation: {e}")
        pred_boxes = np.array([])
        pred_scores = np.array([])
        pred_classes = np.array([])

    try:
        gt_boxes = np.concatenate(all_gt_boxes) if all_gt_boxes and len(all_gt_boxes[0]) > 0 else np.array([])
        gt_classes = np.concatenate(all_gt_classes) if all_gt_classes and len(all_gt_classes[0]) > 0 else np.array([])
    except ValueError as e:
        print(f"Warning in concatenation: {e}")
        gt_boxes = np.array([])
        gt_classes = np.array([])

    mAP, ap_per_class = calculate_mAP(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes)

    for class_id, ap in ap_per_class.items():
        class_name = VOC_CLASSES[class_id+1] if class_id+1 < len(VOC_CLASSES) else f"class_{class_id}"
        print(f"  AP for {class_name}: {ap:.4f}")

    return mAP
def main():
    # Data parameters
    img_size = 416
    batch_size = 8
    subset_size = 400  
    
    voc_root = download_pascal_voc()
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = PascalVOCDataset(voc_root, split='train', transform=transform, subset_size=subset_size)
    val_dataset = PascalVOCDataset(voc_root, split='val', transform=transform, subset_size=subset_size//4)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=lambda x: (torch.stack([item[0] for item in x]), 
                             torch.stack([item[1] for item in x]))
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=lambda x: (torch.stack([item[0] for item in x]), 
                             torch.stack([item[1] for item in x]))
    )
    
    model = ResNet50YOLO(num_classes=NUM_CLASSES, pretrained=True).to(device)
    
    # Define loss function and optimizer
    criterion = YOLOLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Train model
    print("Starting training...")
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs=15
    )
    
    # Load best model
    model.load_state_dict(torch.load("best_model.pth"))
    
    # Final evaluation
    print("Final evaluation:")
    mAP = evaluate_model(model, val_loader)
    print(f"Final mAP: {mAP:.4f}")
    
    # Calculate precision and recall metrics
    print("\nCalculating precision and recall metrics...")
    pred_data = evaluate_precision_recall(model, val_loader)
    all_pred_boxes, all_pred_scores, all_pred_classes, all_gt_boxes, all_gt_classes = pred_data
    
    # Plot PR curves for specific classes 
    class_names_to_plot = ['person', 'car', 'dog']
    class_ids_to_plot = [CLASS_TO_IDX[name] for name in class_names_to_plot if name in CLASS_TO_IDX]
    
    print(f"\nPlotting PR curves for classes: {', '.join(class_names_to_plot)}")
    plot_pr_curve(
        all_pred_boxes, all_pred_scores, all_pred_classes,
        all_gt_boxes, all_gt_classes,
        class_ids=class_ids_to_plot
    )
    
    # Save and download final model
    print("\nSaving final model...")
    final_model_path = save_and_download_model(model, "yolo_resnet_final.pth")
    print(f"Model saved to: {final_model_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_mAP'])
    plt.title('Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Visualize test predictions
    model.eval()
    val_data_iter = iter(val_loader)
    images, targets = next(val_data_iter)
    
    with torch.no_grad():
        predictions = model(images.to(device))
        pred_boxes, pred_scores, pred_classes = decode_predictions(predictions)
        
        for i in range(min(3, len(images))):
            img = images[i].cpu()
            img = transforms.ToPILImage()(img)
            
            boxes = pred_boxes[i].cpu().numpy()
            scores = pred_scores[i].cpu().numpy()
            classes = pred_classes[i].cpu().numpy()
            
            boxes, scores, classes = apply_nms(boxes, scores, classes)
            
            result_img = visualize_predictions(img, boxes, scores, classes)
            
            # Save the result
            result_img.save(f"prediction_sample_{i}.png")
            plt.figure(figsize=(8, 8))
            plt.imshow(result_img)
            plt.axis('off')
            plt.title(f"Detection Sample {i+1}")
            plt.show()

if __name__ == "__main__":
    main()
