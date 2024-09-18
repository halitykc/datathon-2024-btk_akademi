#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
veri = pd.read_csv("test_x.csv")


# In[19]:


veri.head()


# In[20]:


veri.Cinsiyet = veri.Cinsiyet.str.lower()
veri.Cinsiyet.unique()


# In[21]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

sexmap = {"erkek":1,
          "kadın":2,
          "belirtmek istemiyorum":0
          }

veri.Cinsiyet = veri.Cinsiyet.map(sexmap)
veri.Cinsiyet.unique()


# In[22]:


veri.head()


# In[24]:


import re


# 4 basamaklı sayıyı bulup yeni sütuna ekleyen fonksiyon
def yil_bul(veri):
    if isinstance(veri, str):
        # 4 basamaklı yılı bul
        yil = re.findall(r'\b\d{4}\b', veri)
        if yil:
            return yil[0]
        # 2 basamaklı yılı bul
        yil_2_basamakli = re.findall(r'\b\d{2}\b', veri)
        if yil_2_basamakli:
            yil_2_basamakli = yil_2_basamakli[-1]  # Son iki basamaklı yılı al
            # 2 basamaklı yılı 4 basamaklıya dönüştür
            if int(yil_2_basamakli) < 50:
                return '20' + yil_2_basamakli
            else:
                return '19' + yil_2_basamakli
    return None

# 'veriler' sütunundaki 4 basamaklı sayıyı (yılı) 'yil' sütununa ekleme
veri['yil'] = veri['Dogum Tarihi'].apply(yil_bul)
veri["Yas"] = 2024-veri["yil"].astype(float)


# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[35]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=veri, x='Yas')
plt.title('Yaş Sütunu Boxplot')
plt.show()


# In[37]:


q1 = veri['Yas'].quantile(0.25)
q2 = veri['Yas'].quantile(0.50)  # Medyan
q3 = veri['Yas'].quantile(0.75)

print(f"Q1 (1. Çeyrek): {q1}")
print(f"Q2 (Medyan): {q2}")
print(f"Q3 (3. Çeyrek): {q3}")


# In[40]:


IQR = q3 - q1
# Aykırı değer sınırlarını belirleme
lower_bound = q1 - 1.5 * IQR
upper_bound = q3 + 1.5 * IQR
outliers = veri[(veri['Yas'] < lower_bound) | (veri['Yas'] > upper_bound)]

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Aykırı Değerler:")
print(outliers["Yas"])


# In[41]:


veri.loc[veri['Yas'] <= 16.5, "Yas"] = 16.5
veri.loc[veri["Yas"] > 36.5, 'Yas'] = 36.5


# In[42]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=veri, x='Yas')
plt.title('Yaş Sütunu Boxplot')
plt.show()


# In[45]:


#artık Doğum tarihi sütununa ihtiyaç yok
newdf = veri.drop(["Dogum Tarihi"],axis=1)
newdf.head()


# In[46]:


from fuzzywuzzy import process


# Şehir isimleri listesi

# Düzeltme fonksiyonu
def correct_city_name(city_name, city_names):
    if pd.isna(city_name):
        return city_name  # NaN değerleri değiştirmeyin
    closest_match = process.extractOne(city_name, city_names)
    return closest_match[0] if closest_match else city_name

# DataFrame üzerinde uygulama

sehirlerDogru = [
    "Adana", "Adıyaman", "Afyonkarahisar", "Ağrı", "Aksaray", "Amasya", "Ankara", "Antalya",
    "Ardahan", "Artvin", "Aydın", "Balıkesir", "Bartın", "Batman", "Bayburt", "Bilecik", "Bingöl",
    "Bitlis", "Bolu", "Burdur", "Bursa", "Çanakkale", "Çankırı", "Çorum", "Denizli", "Diyarbakır",
    "Düzce", "Edirne", "Elazığ", "Erzincan", "Erzurum", "Eskişehir", "Gaziantep", "Giresun", "Gümüşhane",
    "Hakkari", "Hatay", "Iğdır", "Isparta", "İstanbul", "İzmir", "Kahramanmaraş", "Karabük", "Karaman",
    "Kars", "Kastamonu", "Kayseri", "Kilis", "Kırıkkale", "Kırklareli", "Kırşehir", "Kocaeli", "Konya",
    "Kütahya", "Malatya", "Manisa", "Mardin", "Mersin", "Muğla", "Muş", "Nevşehir", "Niğde", "Ordu",
    "Osmaniye", "Rize", "Sakarya", "Samsun", "Siirt", "Sinop", "Sivas", "Şanlıurfa", "Şırnak", "Tekirdağ",
    "Tokat", "Trabzon", "Tunceli", "Uşak", "Van", "Yalova", "Yozgat", "Zonguldak"
]

newdf['Dogum Yeri'] = newdf['Dogum Yeri'].apply(lambda x: correct_city_name(x, sehirlerDogru))


# In[62]:


yedek = newdf.copy()
yedek["Dogum Yeri"] = yedek["Dogum Yeri"].fillna("Bilinmiyor")
yedek["Dogum Yeri"] = le.fit_transform(yedek["Dogum Yeri"])
print("Eşleme:", dict(zip(le.classes_, le.transform(le.classes_))))


# In[55]:


yedek.head()


# In[56]:


newdf['Ikametgah Sehri'] = newdf['Ikametgah Sehri'].apply(lambda x: correct_city_name(x, sehirlerDogru))


# In[63]:


yedek2 = yedek.copy()
yedek2["Ikametgah Sehri"] = yedek2["Ikametgah Sehri"].fillna("Bilinmiyor")
sehirKarşilik = {'Adana': 0, 'Adıyaman': 1, 'Afyonkarahisar': 2, 'Aksaray': 3, 'Amasya': 4, 'Ankara': 5, 'Antalya': 6, 'Ardahan': 7, 'Artvin': 8, 'Aydın': 9, 'Ağrı': 10, 'Balıkesir': 11, 'Bartın': 12, 'Batman': 13, 'Bayburt': 14, 'Bilecik': 15, 'Bilinmiyor': 16, 'Bingöl': 17, 'Bitlis': 18, 'Bolu': 19, 'Burdur': 20, 'Bursa': 21, 'Denizli': 22, 'Diyarbakır': 23, 'Düzce': 24, 'Edirne': 25, 'Elazığ': 26, 'Erzincan': 27, 'Erzurum': 28, 'Eskişehir': 29, 'Gaziantep': 30, 'Giresun': 31, 'Gümüşhane': 32, 'Hakkari': 33, 'Hatay': 34, 'Isparta': 35, 'Iğdır': 36, 'Kahramanmaraş': 37, 'Karabük': 38, 'Karaman': 39, 'Kars': 40, 'Kastamonu': 41, 'Kayseri': 42, 'Kilis': 43, 'Kocaeli': 44, 'Konya': 45, 'Kütahya': 46, 'Kırklareli': 47, 'Kırıkkale': 48, 'Kırşehir': 49, 'Malatya': 50, 'Manisa': 51, 'Mardin': 52, 'Mersin': 53, 'Muğla': 54, 'Muş': 55, 'Nevşehir': 56, 'Niğde': 57, 'Ordu': 58, 'Osmaniye': 59, 'Rize': 60, 'Sakarya': 61, 'Samsun': 62, 'Siirt': 63, 'Sinop': 64, 'Sivas': 65, 'Tekirdağ': 66, 'Tokat': 67, 'Trabzon': 68, 'Tunceli': 69, 'Uşak': 70, 'Van': 71, 'Yalova': 72, 'Yozgat': 73, 'Zonguldak': 74, 'Çanakkale': 75, 'Çankırı': 76, 'Çorum': 77, 'İstanbul': 78, 'İzmir': 79, 'Şanlıurfa': 80, 'Şırnak': 81}
yedek2["Ikametgah Sehri"] = yedek2["Ikametgah Sehri"].map(sehirKarşilik)


# In[64]:


yedek2.head()


# In[65]:


uniCorrect = ["Adana Alparslan Türkeş Bilim ve Teknoloji Üniversitesi",
"Çukurova Üniversitesi",
"Adıyaman Üniversitesi",
"Afyon Kocatepe Üniversitesi",
"Afyonkarahisar Sağlık Bilimleri Üniversitesi",
"Ağrı İbrahim Çeçen Üniversitesi",
"Aksaray Üniversitesi",
"Amasya Üniversitesi",
"Jandarma ve Sahil Güvenlik Akademisi (Askerî)",
"Ankara Üniversitesi",
"Ankara Müzik ve Güzel Sanatlar Üniversitesi",
"Ankara Hacı Bayram Veli Üniversitesi",
"Ankara Sosyal Bilimler Üniversitesi",
"Gazi Üniversitesi",
"Hacettepe Üniversitesi",
"Orta Doğu Teknik Üniversitesi",
"Ankara Yıldırım Beyazıt Üniversitesi",
"Polis Akademisi",
"Ankara Bilim Üniversitesi",
"Ankara Medipol Üniversitesi",
"Atılım Üniversitesi",
"Başkent Üniversitesi",
"Çankaya Üniversitesi",
"İhsan Doğramacı Bilkent Üniversitesi",
"Lokman Hekim Üniversitesi",
"Ostim Teknik Üniversitesi",
"TED Üniversitesi",
"TOBB Ekonomi ve Teknoloji Üniversitesi",
"Ufuk Üniversitesi",
"Türk Hava Kurumu Üniversitesi",
"Vakıf",
"Akdeniz Üniversitesi",
"Alanya Alaaddin Keykubat Üniversitesi",
"Alanya Üniversitesi",
"Antalya Belek Üniversitesi",
"Antalya Bilim Üniversitesi",
"Ardahan Üniversitesi",
"Artvin Çoruh Üniversitesi",
"Aydın Adnan Menderes Üniversitesi",
"Balıkesir Üniversitesi",
"Bandırma Onyedi Eylül Üniversitesi",
"Bartın Üniversitesi",
"Batman Üniversitesi",
"Bayburt Üniversitesi",
"Bilecik Şeyh Edebali Üniversitesi",
"Bingöl Üniversitesi",
"Bitlis Eren Üniversitesi",
"Bolu Abant İzzet Baysal Üniversitesi",
"Burdur Mehmet Akif Ersoy Üniversitesi",
"Bursa Teknik Üniversitesi",
"Bursa Uludağ Üniversitesi",
"Mudanya Üniversitesi",
"Çanakkale Onsekiz Mart Üniversitesi",
"Çankırı Karatekin Üniversitesi",
"Hitit Üniversitesi",
"Pamukkale Üniversitesi",
"Dicle Üniversitesi",
"Düzce Üniversitesi",
"Trakya Üniversitesi",
"Fırat Üniversitesi",
"Erzincan Binali Yıldırım Üniversitesi",
"Atatürk Üniversitesi",
"Erzurum Teknik Üniversitesi",
"Anadolu Üniversitesi",
"Eskişehir Osmangazi Üniversitesi",
"Eskişehir Teknik Üniversitesi",
"Gaziantep Üniversitesi",
"Gaziantep İslam Bilim ve Teknoloji Üniversitesi",
"Hasan Kalyoncu Üniversitesi",
"Sanko Üniversitesi",
"Giresun Üniversitesi",
"Gümüşhane Üniversitesi",
"Hakkari Üniversitesi",
"İskenderun Teknik Üniversitesi",
"Hatay Mustafa Kemal Üniversitesi",
"Iğdır Üniversitesi",
"Süleyman Demirel Üniversitesi",
"Isparta Uygulamalı Bilimler Üniversitesi",
"Boğaziçi Üniversitesi",
"Galatasaray Üniversitesi",
"İstanbul Medeniyet Üniversitesi",
"İstanbul Teknik Üniversitesi",
"İstanbul Üniversitesi",
"İstanbul Üniversitesi-Cerrahpaşa",
"Marmara Üniversitesi",
"Milli Savunma Üniversitesi (Askerî)",
"Mimar Sinan Güzel Sanatlar Üniversitesi",
"Türk-Alman Üniversitesi",
"Türk-Japon Bilim ve Teknoloji Üniversitesi",
"Sağlık Bilimleri Üniversitesi",
"Yıldız Teknik Üniversitesi",
"Acıbadem Üniversitesi",
"Altınbaş Üniversitesi",
"Bahçeşehir Üniversitesi",
"Beykoz Üniversitesi",
"Bezmialem Vakıf Üniversitesi",
"Biruni Üniversitesi",
"Demiroğlu Bilim Üniversitesi",
"Doğuş Üniversitesi",
"Fatih Sultan Mehmet Üniversitesi",
"Fenerbahçe Üniversitesi",
"Haliç Üniversitesi",
"Işık Üniversitesi",
"İbn Haldun Üniversitesi",
"İstanbul 29 Mayıs Üniversitesi",
"İstanbul Arel Üniversitesi",
"İstanbul Atlas Üniversitesi",
"İstanbul Aydın Üniversitesi",
"İstanbul Beykent Üniversitesi",
"İstanbul Bilgi Üniversitesi",
"İstanbul Esenyurt Üniversitesi",
"İstanbul Galata Üniversitesi",
"İstanbul Gedik Üniversitesi",
"İstanbul Gelişim Üniversitesi",
"İstanbul Kent Üniversitesi",
"İstanbul Kültür Üniversitesi",
"İstanbul Medipol Üniversitesi",
"İstanbul Nişantaşı Üniversitesi",
"İstanbul Okan Üniversitesi",
"İstanbul Rumeli Üniversitesi",
"İstanbul Sabahattin Zaim Üniversitesi",
"İstanbul Sağlık ve Teknoloji Üniversitesi",
"İstanbul Ticaret Üniversitesi",
"İstanbul Topkapı Üniversitesi",
"İstanbul Yeni Yüzyıl Üniversitesi",
"İstinye Üniversitesi",
"Kadir Has Üniversitesi",
"Koç Üniversitesi",
"Maltepe Üniversitesi",
"MEF Üniversitesi",
"Özyeğin Üniversitesi",
"Piri Reis Üniversitesi",
"Sabancı Üniversitesi",
"Üsküdar Üniversitesi",
"Vakıf",
"Dokuz Eylül Üniversitesi",
"Ege Üniversitesi",
"İzmir Yüksek Teknoloji Enstitüsü",
"İzmir Kâtip Çelebi Üniversitesi",
"İzmir Bakırçay Üniversitesi",
"İzmir Demokrasi Üniversitesi",
"İzmir Ekonomi Üniversitesi",
"İzmir Tınaztepe Üniversitesi",
"Yaşar Üniversitesi",
"Kahramanmaraş Sütçü İmam Üniversitesi",
"Kahramanmaraş İstiklal Üniversitesi",
"Karabük Üniversitesi",
"Karamanoğlu Mehmetbey Üniversitesi",
"Kafkas Üniversitesi",
"Kastamonu Üniversitesi",
"Abdullah Gül Üniversitesi",
"Erciyes Üniversitesi",
"Kayseri Üniversitesi",
"Nuh Naci Yazgan Üniversitesi",
"Kırıkkale Üniversitesi",
"Kırklareli Üniversitesi",
"Kırşehir Ahi Evran Üniversitesi",
"Kilis 7 Aralık Üniversitesi",
"Gebze Teknik Üniversitesi",
"Kocaeli Üniversitesi",
"Kocaeli Sağlık ve Teknoloji Üniversitesi",
"Konya Teknik Üniversitesi",
"Necmettin Erbakan Üniversitesi",
"Selçuk Üniversitesi",
"Konya Gıda ve Tarım Üniversitesi",
"KTO Karatay Üniversitesi",
"Kütahya Dumlupınar Üniversitesi",
"Kütahya Sağlık Bilimleri Üniversitesi",
"İnönü Üniversitesi",
"Malatya Turgut Özal Üniversitesi",
"Manisa Celal Bayar Üniversitesi",
"Mardin Artuklu Üniversitesi",
"Mersin Üniversitesi",
"Tarsus Üniversitesi",
"Çağ Üniversitesi",
"Toros Üniversitesi",
"Muğla Sıtkı Koçman Üniversitesi",
"Muş Alparslan Üniversitesi",
"Nevşehir Hacı Bektaş Veli Üniversitesi",
"Kapadokya Üniversitesi",
"Niğde Ömer Halisdemir Üniversitesi",
"Ordu Üniversitesi",
"Osmaniye Korkut Ata Üniversitesi",
"Recep Tayyip Erdoğan Üniversitesi",
"Sakarya Üniversitesi",
"Sakarya Uygulamalı Bilimler Üniversitesi",
"Ondokuz Mayıs Üniversitesi",
"Samsun Üniversitesi",
"Siirt Üniversitesi",
"Sinop Üniversitesi",
"Sivas Cumhuriyet Üniversitesi",
"Sivas Bilim ve Teknoloji Üniversitesi",
"Harran Üniversitesi",
"Şırnak Üniversitesi",
"Tekirdağ Namık Kemal Üniversitesi",
"Tokat Gaziosmanpaşa Üniversitesi",
"Karadeniz Teknik Üniversitesi",
"Trabzon Üniversitesi",
"Avrasya Üniversitesi",
"Munzur Üniversitesi",
"Uşak Üniversitesi",
"Van Yüzüncü Yıl Üniversitesi",
"Yalova Üniversitesi",
"Yozgat Bozok Üniversitesi",
"Zonguldak Bülent Ecevit Üniversitesi"
]
yedek2['Universite Adi'] = yedek2['Universite Adi'].apply(lambda x: correct_city_name(x, uniCorrect))


# # CHECK POINT

# In[184]:


yedek3 = yedek2.copy()
yedek3.head()


# In[185]:


plt.figure(figsize=(10, 6))
sns.heatmap(yedek3.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()


# In[186]:


missing_values = yedek3.isnull().mean() * 100
missing_values = missing_values[missing_values >0]
missing_values = missing_values.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=missing_values.index, y=missing_values.values, palette='cubehelix')
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Percentage of Missing Values')
plt.title('Missing Values Distribution in df_train')
plt.show()


# In[187]:


from dython.nominal import associations

associations_df = associations(yedek3[:10000], nominal_columns='all', plot=False)
corr_matrix = associations_df['corr']
plt.figure(figsize=(20, 8))
plt.gcf().set_facecolor('#FFFDD0') 
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix including Categorical Features')
plt.show()


# In[188]:


yedek3["Universite Turu"] = yedek3["Universite Turu"] .str.lower()
uniTur = {"özel": 1,
          "devlet":0}
yedek3["Universite Turu"] = yedek3["Universite Turu"].map(uniTur)


# In[189]:


yedek3["Burs Aliyor mu?"] = yedek3["Burs Aliyor mu?"] .str.lower()
uniTur = {"evet": 1,
          "hayır":0}
yedek3["Burs Aliyor mu?"] = yedek3["Burs Aliyor mu?"].map(uniTur)


# In[190]:


yedek3["Universite Kacinci Sinif"] = yedek3["Universite Kacinci Sinif"] .str.lower()
uniTur = {"hazırlık": 0,
          "0":0,
          "1":1,
          "2":2,
          "3":3,
          "4":4,
          "5":5,
          "6":6,
          "yüksek lisans":7,
          "tez":7,
          "mezun":8}
yedek3["Universite Kacinci Sinif"] = yedek3["Universite Kacinci Sinif"].map(uniTur)


# In[191]:


yedek3['Universite Not Ortalamasi'] = yedek3['Universite Not Ortalamasi'].str.replace(' ', '', regex=False)
yedek3["Universite Not Ortalamasi"] = yedek3["Universite Not Ortalamasi"].str.lower()

uninot ={"ortalamabulunmuyor":0,
         "notortalamasıyok":0,
         "hazırlığım":0,
         
         "0-1.79":1,
         "1.80-2.49":1,
         "1.00-2.50":1,
         "2.50vealtı":1,
         "2.00-2.50":1,
         
         "3.00-2.50":2,
         "2.50-3.00":2,
         "2.50-2.99":2,
         
         "3.50-3":3,
         "3.00-4.00":3,
         "3.50-4.00":3,
         "3.00-3.50":3,
         "4-3.5":3,
         "4.0-3.5":3,
         "3.00-3.49":3
         }

yedek3["Universite Not Ortalamasi"] = yedek3["Universite Not Ortalamasi"].map(uninot)


# In[192]:


yedek3["Universite Not Ortalamasi"].unique()
           


# In[193]:


pd.set_option('display.max_columns',None)
yedek3 = yedek3.drop(["Bölüm","Lise Adi","Lise Adi Diger","Lise Bolumu","Lise Bolum Diger","Daha Önceden Mezun Olunduysa, Mezun Olunan Üniversite"],axis = 1)
yedek3.head()


# In[194]:


yedek3 = yedek3.drop(["Hangi STK'nin Uyesisiniz?"],axis =1 )


# In[195]:


def eksiveriOrani(veri):
    missing_rate = veri.isnull().sum() / len(veri) * 100
    print(missing_rate[missing_rate > 0].sort_values(ascending=False))

eksiveriOrani(yedek3)


# In[196]:


yedek3["Daha Once Baska Bir Universiteden Mezun Olmus"].unique()


# In[197]:


dahaonceunimezun ={"Hayır":0,
         "Evet":1,
         }

yedek3["Daha Once Baska Bir Universiteden Mezun Olmus"] = yedek3["Daha Once Baska Bir Universiteden Mezun Olmus"].map(dahaonceunimezun)


# In[198]:


yedek3.head()


# In[199]:


yedek3["Lise Turu"] =yedek3["Lise Turu"].str.lower().str.replace(" ","").str.replace("lisesi","").str.replace("lise","")
yedek3["Lise Turu"].unique()


# In[200]:


liseTur = {"diğer":0,
           "devlet":1,
           "anadolu":2,
           "meslek":3,
           "fen":4,
           "özel":5,
           "düz":6,
           "i̇mamhatip":7,
           }
yedek3["Lise Turu"] = yedek3["Lise Turu"].map(liseTur)


# In[201]:


yedek3.head()


# In[202]:


yedek3["Lise Mezuniyet Notu"]
yedek3["Lise Mezuniyet Notu"].unique()


# In[203]:


yedek3['Lise Mezuniyet Notu'] = yedek3['Lise Mezuniyet Notu'].str.replace(' ', '', regex=False)
yedek3["Lise Mezuniyet Notu"] = yedek3["Lise Mezuniyet Notu"].str.lower()
lisenot = {
    "notortalamasıyok":0,
    "0":0,
    "2.50vealtı":1,
    "44-0":1,
    "54-45":1,
    "25-50":1,
    "0-25":1,
    "25-49":1,
    "0-24":1,

    "69-55":2,
    "50-74":2,
    "50-75":2,
    "3.00-2.50":2,

    "3.50-3":3,
    "3.50-3.00":3,

    "3.00-4.00":4,
    "84-70":4,

    "4.00-3.50":5,
    "100-85":5,
    "75-100":5,
    
}
yedek3["Lise Mezuniyet Notu"] = yedek3["Lise Mezuniyet Notu"].map(lisenot)

yedek3["Lise Mezuniyet Notu"].unique()


# In[204]:


yedek3["Baska Bir Kurumdan Burs Aliyor mu?"].unique()


# In[205]:


yedek3["Baska Bir Kurumdan Burs Aliyor mu?"] = yedek3["Baska Bir Kurumdan Burs Aliyor mu?"] .str.lower()
baskayerdenburs = {"evet": 1,
          "hayır":0}
yedek3["Baska Bir Kurumdan Burs Aliyor mu?"] = yedek3["Baska Bir Kurumdan Burs Aliyor mu?"].map(baskayerdenburs)
yedek3["Baska Bir Kurumdan Burs Aliyor mu?"].unique()


# In[206]:


yedek3['Anne Egitim Durumu'] = yedek3['Anne Egitim Durumu'].str.replace(' ', '', regex=False)
yedek3["Anne Egitim Durumu"] = yedek3["Anne Egitim Durumu"].str.lower()
anneegitm ={
    "eğitimiyok":0,
    "eği̇ti̇myok":0,
    "eğitimyok":0,

    "i̇lkokul":1,
    "i̇lkokulmezunu":1,
    
    "ortaokul":2,
    "ortaokulmezunu":2,

    "li̇se":3,
    "lisemezunu":3,
    
    "üni̇versi̇te":4,
    "üniversitemezunu":4,
    "üniversite":4,


    "yüksekli̇sans":5,
    "doktora":5,
    "yükseklisans/doktara":5,
    "yükseklisans/doktora":5


}
yedek3["Anne Egitim Durumu"] = yedek3["Anne Egitim Durumu"].map(anneegitm)
yedek3["Anne Egitim Durumu"].unique()


# In[207]:


yedek3.head()


# In[208]:


yedek3["Anne Calisma Durumu"] = yedek3["Anne Calisma Durumu"] .str.lower()
annecalismadurumu = {"evet": 1,
          "hayır":0,
          "emekli":2}
yedek3["Anne Calisma Durumu"] = yedek3["Anne Calisma Durumu"].map(annecalismadurumu)
yedek3["Anne Calisma Durumu"].unique()


# In[209]:


yedek3["Anne Sektor"] = yedek3["Anne Sektor"] .str.lower()
yedek3["Anne Sektor"].unique()

annesektor = {
    "0":0,
    "-":0,

    "kamu":1,

    "özel sektör":2,

    "di̇ğer":3,
    "diğer":3,
}
yedek3["Anne Sektor"] = yedek3["Anne Sektor"].map(annesektor)
yedek3["Anne Sektor"].unique()


# In[210]:


yedek3['Baba Egitim Durumu'] = yedek3['Baba Egitim Durumu'].str.replace(' ', '', regex=False)
yedek3["Baba Egitim Durumu"] = yedek3["Baba Egitim Durumu"].str.lower()
anneegitm ={
    "eğitimiyok":0,
    "eği̇ti̇myok":0,
    "eğitimyok":0,

    "i̇lkokul":1,
    "i̇lkokulmezunu":1,
    
    "ortaokul":2,
    "ortaokulmezunu":2,

    "li̇se":3,
    "lisemezunu":3,
    
    "üni̇versi̇te":4,
    "üniversitemezunu":4,
    "üniversite":4,


    "yüksekli̇sans":5,
    "doktora":5,
    "yükseklisans/doktara":5,
    "yükseklisans/doktora":5


}
yedek3["Baba Egitim Durumu"] = yedek3["Baba Egitim Durumu"].map(anneegitm)
yedek3["Baba Egitim Durumu"].unique()


# In[211]:


yedek3["Baba Calisma Durumu"] = yedek3["Baba Calisma Durumu"] .str.lower()
Babacalismadurumu = {"evet": 1,
          "hayır":0,
          "emekli":2}
yedek3["Baba Calisma Durumu"] = yedek3["Baba Calisma Durumu"].map(Babacalismadurumu)
yedek3["Baba Calisma Durumu"].unique()


# In[212]:


yedek3["Baba Sektor"] = yedek3["Baba Sektor"].str.lower()

yedek3["Baba Sektor"].unique()


# In[213]:


yedek3["Baba Sektor"] = yedek3["Baba Sektor"] .str.lower()
yedek3["Baba Sektor"].unique()

annesektor = {
    "0":0,
    "-":0,

    "kamu":1,

    "özel sektör":2,

    "di̇ğer":3,
    "diğer":3,
}
yedek3["Baba Sektor"] = yedek3["Baba Sektor"].map(annesektor)
yedek3["Baba Sektor"].unique()


# In[214]:


yedek3["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"].unique()


# In[215]:


yedek3["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"] = yedek3["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"] .str.lower()
grsmKulupUye = {"evet": 1,
          "hayır":0}
yedek3["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"] = yedek3["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"].map(grsmKulupUye)
yedek3["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"].unique()


# In[216]:


yedek3["Uye Oldugunuz Kulubun Ismi"] = le.fit_transform(yedek3["Uye Oldugunuz Kulubun Ismi"])
yedek3.head()


# In[217]:


yedek3["Profesyonel Bir Spor Daliyla Mesgul musunuz?"].unique()


# In[218]:


yedek3["Profesyonel Bir Spor Daliyla Mesgul musunuz?"] = yedek3["Profesyonel Bir Spor Daliyla Mesgul musunuz?"] .str.lower()
grsmKulupUye = {"evet": 1,
          "hayır":0}
yedek3["Profesyonel Bir Spor Daliyla Mesgul musunuz?"] = yedek3["Profesyonel Bir Spor Daliyla Mesgul musunuz?"].map(grsmKulupUye)
yedek3["Profesyonel Bir Spor Daliyla Mesgul musunuz?"].unique()


# In[219]:


yedek3["Spor Dalindaki Rolunuz Nedir?"].unique()


# In[221]:


sporRol = {"0":0,
           "-":0,

           "Kaptan":1,
           "KAPTAN / LİDER":1,
           "Lider/Kaptan":1,
           "KAPTAN / LİDER":1,

           "Takım Oyuncusu":2,

           "Bireysel":3,
           "Bireysel Spor":3,

            "DİĞER":4,
            "Diğer":4,
           }
yedek3["Spor Dalindaki Rolunuz Nedir?"] = yedek3["Spor Dalindaki Rolunuz Nedir?"].map(sporRol)
yedek3["Spor Dalindaki Rolunuz Nedir?"].unique()


# In[222]:


yedek3["Aktif olarak bir STK üyesi misiniz?"].unique()


# In[223]:


yedek3["Aktif olarak bir STK üyesi misiniz?"] = yedek3["Aktif olarak bir STK üyesi misiniz?"] .str.lower()
aktifstkuyesi = {"evet": 1,
          "hayır":0}
yedek3["Aktif olarak bir STK üyesi misiniz?"] = yedek3["Aktif olarak bir STK üyesi misiniz?"].map(aktifstkuyesi)
yedek3["Aktif olarak bir STK üyesi misiniz?"].unique()


# In[224]:


yedek3["Stk Projesine Katildiniz Mi?"].unique()


# In[225]:


yedek3["Stk Projesine Katildiniz Mi?"] = yedek3["Stk Projesine Katildiniz Mi?"] .str.lower()
stkprojekatılm = {"evet": 1,
          "hayır":0}
yedek3["Stk Projesine Katildiniz Mi?"] = yedek3["Stk Projesine Katildiniz Mi?"].map(stkprojekatılm)
yedek3["Stk Projesine Katildiniz Mi?"].unique()


# In[226]:


yedek3.head()


# In[227]:


yedek3["Girisimcilikle Ilgili Deneyiminiz Var Mi?"].unique()


# In[228]:


yedek3["Girisimcilikle Ilgili Deneyiminiz Var Mi?"] = yedek3["Girisimcilikle Ilgili Deneyiminiz Var Mi?"] .str.lower()
grsmdeneym = {"evet": 1,
          "hayır":0}
yedek3["Girisimcilikle Ilgili Deneyiminiz Var Mi?"] = yedek3["Girisimcilikle Ilgili Deneyiminiz Var Mi?"].map(grsmdeneym)
yedek3["Girisimcilikle Ilgili Deneyiminiz Var Mi?"].unique()


# In[229]:


yedek3["Ingilizce Biliyor musunuz?"].unique()


# In[230]:


yedek3["Ingilizce Biliyor musunuz?"] = yedek3["Ingilizce Biliyor musunuz?"] .str.lower()
engVar = {"evet": 1,
          "hayır":0}
yedek3["Ingilizce Biliyor musunuz?"] = yedek3["Ingilizce Biliyor musunuz?"].map(engVar)
yedek3["Ingilizce Biliyor musunuz?"].unique()


# In[231]:


yedek3["Ingilizce Seviyeniz?"].unique()


# In[232]:


engSeviye = {
    "0":0,
    "Başlangıç":1,
    "Orta":2,
    "İleri":3
}
yedek3["Ingilizce Seviyeniz?"] = yedek3["Ingilizce Seviyeniz?"].map(engSeviye)


# In[234]:


yedek3.head()

yedek3["Universite Adi"] = le.fit_transform(yedek3["Universite Adi"])


# In[235]:


yedek3.head()


# In[236]:


yedek3['Lise Sehir'] = yedek3['Lise Sehir'].apply(lambda x: correct_city_name(x, sehirlerDogru))


# In[237]:


yedek3["Lise Sehir"] = yedek3["Lise Sehir"].fillna("Bilinmiyor")
sehirKarşilik = {'Adana': 0, 'Adıyaman': 1, 'Afyonkarahisar': 2, 'Aksaray': 3, 'Amasya': 4, 'Ankara': 5, 'Antalya': 6, 'Ardahan': 7, 'Artvin': 8, 'Aydın': 9, 'Ağrı': 10, 'Balıkesir': 11, 'Bartın': 12, 'Batman': 13, 'Bayburt': 14, 'Bilecik': 15, 'Bilinmiyor': 16, 'Bingöl': 17, 'Bitlis': 18, 'Bolu': 19, 'Burdur': 20, 'Bursa': 21, 'Denizli': 22, 'Diyarbakır': 23, 'Düzce': 24, 'Edirne': 25, 'Elazığ': 26, 'Erzincan': 27, 'Erzurum': 28, 'Eskişehir': 29, 'Gaziantep': 30, 'Giresun': 31, 'Gümüşhane': 32, 'Hakkari': 33, 'Hatay': 34, 'Isparta': 35, 'Iğdır': 36, 'Kahramanmaraş': 37, 'Karabük': 38, 'Karaman': 39, 'Kars': 40, 'Kastamonu': 41, 'Kayseri': 42, 'Kilis': 43, 'Kocaeli': 44, 'Konya': 45, 'Kütahya': 46, 'Kırklareli': 47, 'Kırıkkale': 48, 'Kırşehir': 49, 'Malatya': 50, 'Manisa': 51, 'Mardin': 52, 'Mersin': 53, 'Muğla': 54, 'Muş': 55, 'Nevşehir': 56, 'Niğde': 57, 'Ordu': 58, 'Osmaniye': 59, 'Rize': 60, 'Sakarya': 61, 'Samsun': 62, 'Siirt': 63, 'Sinop': 64, 'Sivas': 65, 'Tekirdağ': 66, 'Tokat': 67, 'Trabzon': 68, 'Tunceli': 69, 'Uşak': 70, 'Van': 71, 'Yalova': 72, 'Yozgat': 73, 'Zonguldak': 74, 'Çanakkale': 75, 'Çankırı': 76, 'Çorum': 77, 'İstanbul': 78, 'İzmir': 79, 'Şanlıurfa': 80, 'Şırnak': 81}
yedek3["Lise Sehir"] = yedek3["Lise Sehir"].map(sehirKarşilik)


# In[238]:


yedek3.head()


# In[239]:


yedek3["Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?"] = yedek3["Girisimcilikle Ilgili Deneyiminizi Aciklayabilir misiniz?"].notna().astype(int)


# In[240]:


yedek3.head()


# In[243]:


yedek3 = yedek3.drop(["id"],axis = 1)


# In[246]:


yedek3 = yedek3.drop(["Burs Aldigi Baska Kurum"],axis = 1)


# In[249]:


yedek3 = yedek3.drop(["Baska Kurumdan Aldigi Burs Miktari"],axis = 1)


# In[251]:


yedek3 = yedek3.drop(["Kardes Sayisi"],axis = 1)


# In[253]:


yedek3.corr().sort_values(by="Degerlendirme Puani", ascending=False)


# In[255]:


yedek3.isna().sum().sort_values()


# In[256]:


yedek3["Burslu ise Burs Yuzdesi"] = yedek3["Burslu ise Burs Yuzdesi"].fillna(0)
yedek3["Daha Once Baska Bir Universiteden Mezun Olmus"] = yedek3["Daha Once Baska Bir Universiteden Mezun Olmus"].fillna(0)
yedek3["Ingilizce Seviyeniz?"] = yedek3["Ingilizce Seviyeniz?"].fillna(0)
yedek3["Anne Sektor"] = yedek3["Anne Sektor"].fillna(0)
yedek3["Spor Dalindaki Rolunuz Nedir?"] = yedek3["Spor Dalindaki Rolunuz Nedir?"].fillna(0)

yedek3["Stk Projesine Katildiniz Mi?"] = yedek3["Stk Projesine Katildiniz Mi?"].fillna(0)

yedek3["Baba Sektor"] = yedek3["Baba Sektor"].fillna(0)
yedek3["Aktif olarak bir STK üyesi misiniz?"] = yedek3["Aktif olarak bir STK üyesi misiniz?"].fillna(0)
yedek3["Girisimcilikle Ilgili Deneyiminiz Var Mi?"] = yedek3["Girisimcilikle Ilgili Deneyiminiz Var Mi?"].fillna(0)
yedek3["Baba Calisma Durumu"] = yedek3["Baba Calisma Durumu"].fillna(0)
yedek3["Anne Calisma Durumu"] = yedek3["Anne Calisma Durumu"].fillna(0)
yedek3["Baba Egitim Durumu"] = yedek3["Baba Egitim Durumu"].fillna(0)
yedek3["Anne Egitim Durumu"] = yedek3["Anne Egitim Durumu"].fillna(0)
yedek3["Ingilizce Biliyor musunuz?"] = yedek3["Ingilizce Biliyor musunuz?"].fillna(0)
yedek3["Anne Egitim Durumu"] = yedek3["Anne Egitim Durumu"].fillna(0)

yedek3["Lise Mezuniyet Notu"] = yedek3["Lise Mezuniyet Notu"].fillna(0)

yedek3["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"] = yedek3["Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"].fillna(0)
yedek3["Universite Not Ortalamasi"] = yedek3["Universite Not Ortalamasi"].fillna(0)
yedek3["Profesyonel Bir Spor Daliyla Mesgul musunuz?"] = yedek3["Profesyonel Bir Spor Daliyla Mesgul musunuz?"].fillna(0)
yedek3["Lise Turu"] = yedek3["Lise Turu"].fillna(0)

yedek3["Universite Kacinci Sinif"] = yedek3["Universite Kacinci Sinif"].fillna(0)


# In[257]:


yedek3.isna().sum().sort_values()


# In[265]:


eksiveriOrani(yedek3)
yedek3.shape


# In[ ]:


test = yedek3

