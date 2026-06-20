import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = {
    'bulan': [1,2,3,4,5,6,7,8,9,10],
    'penjualan': [100,120,135,150,160,180,200,220,240,260]
}
df = pd.DataFrame(data)

X = df[['bulan']]
y = df['penjualan']

model = LinearRegression()
model.fit(X, y)

prediksi = model.predict([[11], [12]])
print("Prediksi penjualan bulan ke-11:", round(prediksi[0], 2))
print("Prediksi penjualan bulan ke-12:", round(prediksi[1], 2))

print("Koefisien (b):", round(model.coef_[0], 2))
print("Intercept (a):", round(model.intercept_, 2))

y_pred = model.predict(X)
akurasi = r2_score(y, y_pred)
print("Nilai Akurasi (R²):", round(akurasi, 4))

plt.scatter(X, y, color='blue', label='Data Asli')
plt.plot(X, model.predict(X), color='red', label='Garis Regresi')
plt.xlabel('Bulan')
plt.ylabel('Penjualan')
plt.title('Hubungan Bulan dan Penjualan')
plt.legend()
plt.grid(True)
plt.show()