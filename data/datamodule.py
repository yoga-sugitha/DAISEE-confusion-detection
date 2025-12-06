"""
Lightning DataModule for DAiSEE dataset
Supports both binary and multi-class classification with pre-split data
"""
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pathlib import Path
from .dataset import ImagePathDataset

class DAiSEEDataModule(L.LightningDataModule):
    """
    Lightning DataModule for DAiSEE Confusion Detection
    Works with pre-split Train/Validation/Test directories
    Supports binary (confused vs not_confused) and multi-class (4 levels)
    
    Args:
        data_dir: Path to dataset root (contains Train/Validation/Test folders)
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        img_size: Image size for resizing
        seed: Random seed (kept for compatibility)
        task_type: 'binary' or 'multiclass'
        binary_mapping: 'c0_vs_rest' (not_confused=0, others=1) or 'custom'
        binary_class_map: Custom mapping dict (optional)
    """
    def __init__(
        self, 
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        img_size: int = 224,
        seed: int = 42,
        task_type: str = 'multiclass',
        binary_mapping: str = 'c0_vs_rest',
        binary_class_map: dict = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.seed = seed
        self.task_type = task_type.lower()
        self.binary_mapping = binary_mapping
        self.binary_class_map = binary_class_map
        
        # Set num_classes and class_names
        self.num_classes = 2 if self.task_type == 'binary' else 4
        self.class_names = (
            ['Not Confused', 'Confused'] if self.task_type == 'binary' else
            ['Not Confused', 'Slightly Confused', 'Confused', 'Very Confused']
        )
        
        # Validate task type
        if self.task_type not in ['binary', 'multiclass']:
            raise ValueError(f"task_type must be 'binary' or 'multiclass', got '{task_type}'")
        
        # Verify directories exist
        self.train_dir = self.data_dir / "Train"
        self.val_dir = self.data_dir / "Validation"
        self.test_dir = self.data_dir / "Test"
        
        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            if not split_dir.exists():
                raise ValueError(f"Directory not found: {split_dir}")
        
        # Define transforms (your existing pipeline)
        self.test_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_transform = T.Compose([
            T.Resize((img_size, img_size)),
            # Lighting augmentations
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.2),
            T.RandomApply([T.RandomGrayscale(p=1.0)], p=0.1),
            # Geometric augmentations (mild)
            T.RandomAffine(
                degrees=3,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
            T.RandomPerspective(distortion_scale=0.05, p=0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Dataset placeholders
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """
        Setup datasets from pre-split directories
        Converts to binary labels if needed
        """
        if stage == "fit" or stage is None:
            if self.train_dataset is None:
                train_paths, train_labels = self._load_split(self.train_dir)
                self.train_dataset = ImagePathDataset(
                    train_paths, 
                    train_labels, 
                    transform=self.train_transform
                )
                print(f"\n✓ Loaded training set: {len(self.train_dataset)} samples")
                
            if self.val_dataset is None:
                val_paths, val_labels = self._load_split(self.val_dir)
                self.val_dataset = ImagePathDataset(
                    val_paths, 
                    val_labels, 
                    transform=self.test_transform
                )
                print(f"✓ Loaded validation set: {len(self.val_dataset)} samples")
                
                if self.task_type == 'binary':
                    print(f"  Binary distribution - Class 0: {val_labels.count(0)}, Class 1: {val_labels.count(1)}")
        
        if stage == "test" or stage is None:
            if self.test_dataset is None:
                test_paths, test_labels = self._load_split(self.test_dir)
                self._test_paths = test_paths
                self._test_labels = test_labels
                self.test_dataset = ImagePathDataset(
                    test_paths, 
                    test_labels, 
                    transform=self.test_transform
                )
                print(f"✓ Loaded test set: {len(self.test_dataset)} samples")
                
                if self.task_type == 'binary':
                    print(f"  Binary distribution - Class 0: {test_labels.count(0)}, Class 1: {test_labels.count(1)}")
    
    def _load_split(self, split_dir: Path):
        """
        Load paths and labels from a split directory
        Handles conversion to binary labels if needed
        
        Returns:
            paths: List of image paths
            labels: List of labels (converted to binary if needed)
        """
        paths = []
        labels = []
        
        # Mapping from folder name to class index
        # Folders: 0_not_confused, 1_slightly_confused, 2_confused, 3_very_confused
        folder_to_class = {
            '0_not_confused': 0,
            '1_slightly_confused': 1,
            '2_confused': 2,
            '3_very_confused': 3
        }
        
        # Collect all images
        for folder_name, class_idx in folder_to_class.items():
            folder_path = split_dir / folder_name
            if not folder_path.exists():
                continue
                
            for img_path in folder_path.glob('*.jpg'):  # adjust extension if needed
                paths.append(str(img_path))
                labels.append(class_idx)
        
        # Convert to binary if needed
        if self.task_type == 'binary':
            labels = self._convert_to_binary(labels)
        
        return paths, labels
    
    def get_test_sample(self, idx: int):
        """
        Return (image: Tensor, label: int) for test set index `idx`.
        
        The image is returned **as transformed by test_transform** (normalized, resized).
        This matches exactly what your test_dataloader returns.
        
        Args:
            idx: Index in the test split (0 to len(test_dataset)-1)
            
        Returns:
            (image: torch.Tensor in [C, H, W], label: int)
        """
        if not hasattr(self, '_test_paths') or not hasattr(self, '_test_labels'):
            # Ensure test split exists
            self.setup(stage='test')
        
        if idx < 0 or idx >= len(self._test_paths):
            raise IndexError(f"Test set index {idx} out of range (0–{len(self._test_paths)-1})")
        
        path = self._test_paths[idx]
        label = self._test_labels[idx]
        
        # Load and transform using the same logic as ImagePathDataset
        from PIL import Image
        image = Image.open(path).convert('RGB')
        image = self.test_transform(image)  # Apply test-time transform (normalize, etc.)
        
        return image, label

    
    def _convert_to_binary(self, labels):
        """
        Convert multi-class labels to binary
        
        Args:
            labels: List of original class labels (0-3)
            
        Returns:
            List of binary labels (0 or 1)
        """
        if self.binary_mapping == 'c0_vs_rest':
            # Class 0 (not_confused) = 0, rest (confused states) = 1
            return [0 if label == 0 else 1 for label in labels]
        
        elif self.binary_mapping == 'custom':
            if self.binary_class_map is None:
                raise ValueError("binary_mapping='custom' requires binary_class_map")
            return [self.binary_class_map.get(label, label) for label in labels]
        
        else:
            raise ValueError(
                f"Unknown binary_mapping: {self.binary_mapping}. "
                f"Use 'c0_vs_rest' or 'custom'"
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True if self.num_workers > 0 else False,
            persistent_workers=True if self.num_workers > 0 else False,
        )