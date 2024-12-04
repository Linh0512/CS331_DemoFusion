# %reset -f
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Input, concatenate
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, accuracy_score

# Load the CSV files into pandas DataFrames
print("Loading data...")
eye_tracking_df = pd.read_csv('EyeTracking.csv')
gsr_df = pd.read_csv('GSR.csv')
ecg_df = pd.read_csv('ECG.csv')

# Ensure consistent column names for labels
print("Ensuring consistent column names...")
labels_eye = eye_tracking_df['Quad_Cat'].fillna(method='ffill')
labels_gsr = gsr_df['Quad_Cat'].fillna(method='ffill')
labels_ecg = ecg_df['Quad_Cat'].fillna(method='ffill')

# Drop the label column from the feature set
features_eye = eye_tracking_df.drop(columns=['Quad_Cat'])
features_gsr = gsr_df.drop(columns=['Quad_Cat'])
features_ecg = ecg_df.drop(columns=['Quad_Cat'])

# Padding DataFrames to the same number of rows (using NaN where data is missing)
max_rows = max(len(features_eye), len(features_gsr), len(features_ecg))

# Reindex each DataFrame to have the same number of rows
features_eye = features_eye.reindex(range(max_rows), fill_value=np.nan)
features_gsr = features_gsr.reindex(range(max_rows), fill_value=np.nan)
features_ecg = features_ecg.reindex(range(max_rows), fill_value=np.nan)

# Reindex the labels to match the features
labels_eye = labels_eye.reindex(range(max_rows), fill_value=labels_eye.mode()[0])
labels_gsr = labels_gsr.reindex(range(max_rows), fill_value=labels_gsr.mode()[0])
labels_ecg = labels_ecg.reindex(range(max_rows), fill_value=labels_ecg.mode()[0])

# Impute missing values
print("Imputing missing values...")
imputer = SimpleImputer(strategy='mean')
features_eye_imputed = pd.DataFrame(imputer.fit_transform(features_eye), columns=features_eye.columns)
features_gsr_imputed = pd.DataFrame(imputer.fit_transform(features_gsr), columns=features_gsr.columns)
features_ecg_imputed = pd.DataFrame(imputer.fit_transform(features_ecg), columns=features_ecg.columns)

# Standardize the features
print("Standardizing features...")
scaler = StandardScaler()
features_eye_standardized = pd.DataFrame(scaler.fit_transform(features_eye_imputed), columns=features_eye_imputed.columns)
features_gsr_standardized = pd.DataFrame(scaler.fit_transform(features_gsr_imputed), columns=features_gsr_imputed.columns)
features_ecg_standardized = pd.DataFrame(scaler.fit_transform(features_ecg_imputed), columns=features_ecg_imputed.columns)

# Convert labels to categorical (one-hot encoding)
labels = to_categorical(labels_eye)

# Perform train-test split for each modality
X_train_eye, X_test_eye, y_train, y_test = train_test_split(features_eye_standardized, labels, test_size=0.2, random_state=42)
X_train_gsr, X_test_gsr, _, _ = train_test_split(features_gsr_standardized, labels, test_size=0.2, random_state=42)
X_train_ecg, X_test_ecg, _, _ = train_test_split(features_ecg_standardized, labels, test_size=0.2, random_state=42)

# Use np.expand_dims to add the extra dimension needed for CNN
X_train_eye = np.expand_dims(X_train_eye, axis=2)
X_test_eye = np.expand_dims(X_test_eye, axis=2)
X_train_gsr = np.expand_dims(X_train_gsr, axis=2)
X_test_gsr = np.expand_dims(X_test_gsr, axis=2)
X_train_ecg = np.expand_dims(X_train_ecg, axis=2)
X_test_ecg = np.expand_dims(X_test_ecg, axis=2)

# Build CNN models for each modality
def build_model(input_shape):
    input_layer = Input(shape=input_shape)
    cnn = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Flatten()(cnn)
    output_layer = Dense(y_train.shape[1], activation='softmax')(cnn)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and train models
model_eye = build_model((X_train_eye.shape[1], 1))
model_gsr = build_model((X_train_gsr.shape[1], 1))
model_ecg = build_model((X_train_ecg.shape[1], 1))

# Tính class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(np.argmax(y_train, axis=1)),
    y=np.argmax(y_train, axis=1))

# Chuyển class_weights thành dictionary
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Train the model for each modalities
# Training the eye tracking model
print("Training the Eye Tracking Model...")
model_eye.fit(X_train_eye, y_train, epochs=10, batch_size=32, validation_split=0.2, class_weight=class_weights_dict)

# Training the GSR model
print("Training the GSR Model...")
model_gsr.fit(X_train_gsr, y_train, epochs=10, batch_size=32, validation_split=0.2, class_weight=class_weights_dict)

# Training the ECG model
print("Training the ECG Model...")
model_ecg.fit(X_train_ecg, y_train, epochs=10, batch_size=32, validation_split=0.2, class_weight=class_weights_dict)

# Evaluate the model for each modalities
# Evaluate the eye tracking model
print("Evaluating the Eye Tracking Model...")
test_loss_eye, test_accuracy_eye = model_eye.evaluate(X_test_eye, y_test)
print(f"Eye Tracking Test Accuracy: {test_accuracy_eye:.4f}")

# Evaluate the GSR model
print("Evaluating the GSR Model...")
test_loss_gsr, test_accuracy_gsr = model_gsr.evaluate(X_test_gsr, y_test)
print(f"GSR Test Accuracy: {test_accuracy_gsr:.4f}")

# Evaluate the ECG model
print("Evaluating the ECG Model...")
test_loss_ecg, test_accuracy_ecg = model_ecg.evaluate(X_test_ecg, y_test)
print(f"ECG Test Accuracy: {test_accuracy_ecg:.4f}")

def predict(model, data):
    return model(data)

# Make predictions for each modality
print("Making predictions for each modality...")
# Extract the class with the highest probability
# Predict for eye tracking
y_pred_eye = predict(model_eye, X_test_eye)
y_pred_eye_classes = np.argmax(y_pred_eye, axis=1)

# Predict for GSR
y_pred_gsr = predict(model_gsr, X_test_gsr)
y_pred_gsr_classes = np.argmax(y_pred_gsr, axis=1)

# Predict for ECG
y_pred_ecg = predict(model_ecg, X_test_ecg)
y_pred_ecg_classes = np.argmax(y_pred_ecg, axis=1)


# Compute the accuracy for each model
acc_eye = accuracy_score(np.argmax(y_test, axis=1), y_pred_eye_classes)
acc_gsr = accuracy_score(np.argmax(y_test, axis=1), y_pred_gsr_classes)
acc_ecg = accuracy_score(np.argmax(y_test, axis=1), y_pred_ecg_classes)

# Print the accuracy for each model
print(f"Accuracy for Eye Model: {acc_eye:.4f}")
print(f"Accuracy for GSR Model: {acc_gsr:.4f}")
print(f"Accuracy for ECG Model: {acc_ecg:.4f}")

# Create a list of models and their accuracies
accuracies = [(acc_eye, y_pred_eye_classes), (acc_gsr, y_pred_gsr_classes), (acc_ecg, y_pred_ecg_classes)]

# Sort the models by accuracy (descending order)
accuracies.sort(reverse=True, key=lambda x: x[0])

# Combine the predictions from the best model (highest accuracy)
final_predictions = accuracies[0][1]  # Use the predictions from the model with the highest accuracy

# Print classification report
print("\nClassification Report:")
print(classification_report(np.argmax(y_test, axis=1), final_predictions))

# Calculate and print the confusion matrix
print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), final_predictions)
print(conf_matrix)

# Calculate MCC
mcc = matthews_corrcoef(np.argmax(y_test, axis=1), final_predictions)
print(f"\nMatthews Correlation Coefficient (MCC): {mcc:.4f}")

# Calculate F1 score (macro and weighted)
f1_macro = f1_score(np.argmax(y_test, axis=1), final_predictions, average='macro')
f1_weighted = f1_score(np.argmax(y_test, axis=1), final_predictions, average='weighted')
print(f"\nF1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")

# Calculate balanced accuracy
balanced_acc = balanced_accuracy_score(np.argmax(y_test, axis=1), final_predictions)
print(f"\nBalanced Accuracy: {balanced_acc:.4f}")

# Save the final fused data (train + test) as a single CSV
fused_train_data = pd.DataFrame(np.concatenate([X_train_eye.squeeze(), X_train_gsr.squeeze(), X_train_ecg.squeeze()], axis=1))
fused_train_data['Label'] = np.argmax(y_train, axis=1)

fused_test_data = pd.DataFrame(np.concatenate([X_test_eye.squeeze(), X_test_gsr.squeeze(), X_test_ecg.squeeze()], axis=1))
fused_test_data['Label'] = np.argmax(y_test, axis=1)

# Save to CSV files
fused_train_data.to_csv('fused_train_data.csv', index=False)
fused_test_data.to_csv('fused_test_data.csv', index=False)

print("Final model evaluations and CSV export completed.")
