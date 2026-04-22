# I models, csv, target, instructions per model [cost, etc...]
# O trained models, text log of hyperparameters & performance

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
