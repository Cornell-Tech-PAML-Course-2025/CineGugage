import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set up CNN
backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
backbone.fc = torch.nn.Identity()
backbone.to(DEVICE).eval()

for p in backbone.parameters():
  p.requires_grad = False

img_tf = T.Compose([
    T.ToTensor(),
    T.Resize(224),
    T.Normalize(mean = [0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def embed_image(np_img):
        with torch.no_grad():
            t = img_tf(Image.fromarray((np_img*255).astype("uint8"))).unsqueeze(0).to(DEVICE)
            return backbone(t).cpu().numpy().squeeze()

def prepare_model_inputs(image_data, genre_data):
    visual_embeds = np.stack([embed_image(img) for img in tqdm(image_data, desc="CNN Embedding")])

    X = np.hstack([visual_embeds,genre_data])
    return X

def split_dataset(X, y):    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test

def build_train_ridge_model(x_train, y_train):
    pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=True)),
        ('pca', PCA(n_components=0.95,random_state=42)),
        ('ridge', Ridge()),
    ])

    param_grid = {
        "pca__n_components": [0.90,0.92,0.95,0.98],
        "ridge__alpha":np.logspace(1,5,9),
    }

    cross_validator = KFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(estimator=pipe,
                          param_grid=param_grid,
                          cv=cross_validator,
                          scoring="neg_mean_absolute_error",
                          n_jobs=-1,
                          verbose=2)

    search.fit(x_train, y_train)
    ridge_model = search.best_estimator_
    print("Best α →", search.best_params_["ridge__alpha"])
    return ridge_model

def make_loader(X_arr,y_arr,batch=64,shuffle=True):
  X_t = torch.tensor(X_arr,dtype=torch.float32)
  y_t = torch.tensor(y_arr,dtype=torch.float32).unsqueeze(1)
  return DataLoader(TensorDataset(X_t,y_t),batch_size=batch,shuffle=shuffle)

def get_loaders(x_train, y_train, x_val, y_val, x_test, y_test):
  # Create DataLoader objects for training, validation, and test sets
  train_loader = make_loader(x_train, y_train)
  val_loader = make_loader(x_val, y_val, shuffle=False)
  test_loader = make_loader(x_test, y_test, shuffle=False)
  return train_loader, val_loader, test_loader