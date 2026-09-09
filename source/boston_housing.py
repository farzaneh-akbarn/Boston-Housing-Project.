# پروژه شماره 2: پیش‌بینی قیمت مسکن بوستون
# نام: فرزانه اکبرنژاد

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import joblib

print("شروع پروژه مسکن بوستون")
print("=" * 40)

# بارگذاری داده
ستون‌ها = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
           'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

داده = pd.read_csv('housing.data', sep=r'\s+', names=ستون‌ها)

print("\nداده با موفقیت بارگذاری شد!")
print(f"تعداد رکوردها: {len(داده)}")

# بررسی داده
print("\n--- 5 سطر اول داده ---")
print(داده.head())

print("\n--- اطلاعات کلی داده ---")
(داده.info())

print("\n--- آمار توصیفی ---")
print(داده.describe())

# پاک‌سازی داده
print("\n--- بررسی داده‌های گمشده ---")
print(داده.isnull().sum())
if داده.isnull().sum().sum() == 0:
    print("هیچ داده گمشده‌ای وجود ندارد!")

# رسم نمودارها ( 3 تا)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(داده['MEDV'], bins=30, color='blue', edgecolor='black')
plt.title('نمودار 1: توزیع قیمت مسکن')
plt.xlabel('قیمت (هزار دلار)')
plt.ylabel('تعداد محله')

plt.subplot(1, 3, 2)
plt.scatter(داده['RM'], داده['MEDV'], alpha=0.5)
plt.title('نمودار 2: رابطه تعداد اتاق و قیمت مسکن')
plt.xlabel('میانگین تعداد اتاق')
plt.ylabel('قیمت (هزار دلار)')

plt.subplot(1, 3, 3)
داده.boxplot(column='MEDV', by='CHAS', ax=plt.gca())
plt.title('نمودار 3: قیمت بر اساس مجاورت با رودخانه')
plt.suptitle('')
plt.xlabel('مجاورت با رودخانه (0=خیر، 1=بله)')
plt.ylabel('قیمت (هزار دلار)')

plt.tight_layout()
plt.savefig('نمودارها.png')
plt.show()

# تقسیم داده به آموزش و تست
x = داده.drop('MEDV', axis=1)
y = داده['MEDV']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"\nتعداد داده‌های آموزش: {len(x_train)}")
print(f"تعداد داده‌های تست: {len(x_test)}")

# ساخت مدل درخت تصمیم
مدل = DecisionTreeRegressor(max_depth=8, random_state=42)
مدل.fit(x_train, y_train)

print("\nمدل با موفقیت آموزش دید!")

# ارزیابی مدل
y_pred = مدل.predict(x_test)
خطا = mean_absolute_error(y_test, y_pred)

print(f"\nخطای مدل (MAE): {خطا:.2f} هزار دلار")
if خطا < 4.0:
    print("✅ خطای مدل کمتر از 4.0 است! قبول شد!")
else:
    print(f"⚠️ خطا {خطا:.2f} هست و باید کمتر از 4.0 باشد.")

# اهمیت ویژگی‌ها
اهمیت = مدل.feature_importances_
print("\n--- اهمیت ویژگی‌ها ---")
for i in range(len(x.columns)):
    print(f"{x.columns[i]}: {اهمیت[i]:.4f}")

plt.figure(figsize=(10, 6))
plt.barh(x.columns, اهمیت, color='green')
plt.xlabel('اهمیت')
plt.title('نمودار اهمیت ویژگی‌ها')
plt.tight_layout()
plt.savefig('اهمیت_ویژگی‌ها.png')
plt.show()

بیشترین_اهمیت = 0
نام_ویژگی = ""
for i in range(len(x.columns)):
    if اهمیت[i] > بیشترین_اهمیت:
        بیشترین_اهمیت = اهمیت[i]
        نام_ویژگی = x.columns[i]

print(f"\nمهم‌ترین ویژگی: {نام_ویژگی} با اهمیت {بیشترین_اهمیت:.4f}")

# ذخیره مدل
joblib.dump(مدل, 'مدل_مسکن_بوستون.joblib')
print("\nمدل در فایل 'مدل_مسکن_بوستون.joblib' ذخیره شد!")

# جمع‌بندی
print("\n" + "=" * 40)
print("پروژه با موفقیت انجام شد!")
print(f"خطای نهایی: {خطا:.2f} هزار دلار")
print("فایل‌های تولید شده:")
print("1. boston_housing.py (کد اصلی)")
print("2. نمودارها.png (نمودارهای درخواستی)")
print("3. اهمیت_ویژگی‌ها.png (نمودار اهمیت)")
print("4. مدل_مسکن_بوستون.joblib (مدل ذخیره شده)")
print("=" * 40)