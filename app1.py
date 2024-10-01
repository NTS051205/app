import pandas as pd  # Nhập pandas để xử lý file CSV
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader

# 1. Đọc file CSV từ đường dẫn được cung cấp
csv_file_path = 'F:/Desktop/thuật toán phân tích cảm xúc/texts.csv' 
df = pd.read_csv(csv_file_path)

# Kiểm tra dữ liệu
print("Dữ liệu ban đầu:")
print(df.head())

# 2. Tiền xử lý dữ liệu (PhoBERT đã được tối ưu hóa cho tiếng Việt nên không cần xử lý nhiều)
df['text'] = df['text'].str.lower()

# 3. Sử dụng PhoBERT để tạo Sentence Embedding
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")

def get_phobert_embedding(text):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Áp dụng hàm embedding cho tất cả các văn bản
df['embedding'] = df['text'].apply(get_phobert_embedding)

# 4. Xây dựng Dataset và DataLoader
class TextDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx]), torch.tensor(self.labels[idx])

# Chuyển đổi dữ liệu
X = torch.tensor(list(df['embedding'].values))
y = torch.tensor(df['label'].values)  # Giả sử cột nhãn có tên là 'label'

dataset = TextDataset(X, y)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 5. Định nghĩa mô hình Bi-LSTM
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Khởi tạo mô hình, loss function và optimizer
model = BiLSTM(input_size=768, hidden_size=128, output_size=3)  # 3 lớp cảm xúc (Tích cực, Tiêu cực, Trung lập)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Huấn luyện mô hình
for epoch in range(5):  # Huấn luyện trong 5 epoch
    for embeddings, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 7. Đánh giá mô hình
y_pred = []
y_true = []

model.eval()
with torch.no_grad():
    for embeddings, labels in dataloader:
        outputs = model(embeddings.float())
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(labels.numpy())

print("\nEvaluation Report:")
print(classification_report(y_true, y_pred))

# 8. Dự đoán cảm xúc của văn bản mới (Tiếng Việt)
def predict_sentiment(text):
    embedding = get_phobert_embedding(text)
    with torch.no_grad():
        output = model(torch.tensor([embedding]).float())
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Ví dụ dự đoán
text_example = "Sản phẩm này thật tuyệt vời!"
predicted_class = predict_sentiment(text_example)
sentiment_dict = {0: "Tiêu cực", 1: "Tích cực", 2: "Trung lập"}

print(f'\nInput Text: "{text_example}"')
print(f'Predicted Sentiment: {sentiment_dict[predicted_class]}')
