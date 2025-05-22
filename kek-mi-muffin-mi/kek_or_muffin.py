import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Cupcakes vs Muffins.csv")  

print(df.head())
print(df['Type'].value_counts())  # kac tane kek kac tane muffin var

#  Özellikler ve Etiketler

X = df[['Flour', 'Sugar']]
y_raw = df['Type']

#  Etiketleri Sayıya Çevir
le = LabelEncoder()
y = le.fit_transform(y_raw)  # Cupcake -> 0, Muffin -> 1 

# Modeli Oluştur ve Eğit

model = LogisticRegression()
model.fit(X, y)

# Tahmin Örneği 

yeni_veri = pd.DataFrame([[50, 30]], columns=['Flour', 'Sugar'])  # un miktari 50 olan ve seker miktari 30 olan bir urunun kek mi yoksa muffin mi oldugunu tahmin edelim
tahmin_sayisal = model.predict(yeni_veri)[0]
tahmin_isim = le.inverse_transform([tahmin_sayisal])[0]
print("Tahmin edilen sınıf:", tahmin_isim)


# Karar Sınırı Grafiği Çizimi

sns.scatterplot(data=df, x='Flour', y='Sugar', hue='Type')

x_min, x_max = X['Flour'].min() - 5, X['Flour'].max() + 5
y_min, y_max = X['Sugar'].min() - 5, X['Sugar'].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 1))

grid = np.c_[xx.ravel(), yy.ravel()]
grid_df = pd.DataFrame(grid, columns=['Flour', 'Sugar'])
Z = model.predict(grid_df)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='Pastel1')
plt.title("Karar Sınırı: Kek mi Muffin mi?")
plt.xlabel("Un (Flour)")
plt.ylabel("Şeker (Sugar)")
plt.show()


# Modeli Kaydet
import pickle

with open("kek_muffin_modeli.pkl", "wb") as f:
    pickle.dump((model, le), f)
print("Model başarıyla kaydedildi.")


# Modeli Yükleyip Kullanmak

with open("kek_muffin_modeli.pkl", "rb") as f:
    yuklu_model, yuklu_le = pickle.load(f)

yeni_veri_2 = pd.DataFrame([[60, 20]], columns=['Flour', 'Sugar'])
tahmin_2 = yuklu_model.predict(yeni_veri_2)[0]
print("Yüklenen model tahmini:", yuklu_le.inverse_transform([tahmin_2])[0])
