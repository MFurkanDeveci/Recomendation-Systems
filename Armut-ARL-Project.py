#########################
# GÖREV 1: Veriyi Hazırlama
#########################
#!pip install mlxtend
import pandas as pd
import datetime as dt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules
# Adım 1: armut_data.csv dosyasınız okutunuz.

df = pd.read_csv("ArmutARL-221114-234936/armut_data.csv")
df.head()

df.describe().T
df.isnull().sum()
df.shape

# Adım 2: ServisID her bir CategoryID özelinde farklı bir hizmeti temsil etmektedir.
# ServiceID ve CategoryID'yi "_" ile birleştirerek hizmetleri temsil edecek yeni bir değişken oluşturunuz.

df.ServiceId = df["ServiceId"].astype(str)
df.CategoryId = df["CategoryId"].astype(str)

df["Hizmet"] = (df["ServiceId"] + "_" + df["CategoryId"])
df["Hizmet"].head()

# Adım 3: Veri seti hizmetlerin alındığı tarih ve saatten oluşmaktadır,
# herhangi bir sepet tanımı (fatura vb.) bulunmamaktadır.
# Association Rule Learning uygulayabilmek için bir sepet (fatura vb.) tanımı oluşturulması gerekmektedir.
# Burada sepet tanımı her bir müşterinin aylık aldığı hizmetlerdir.
# Örneğin; 7256 id'li müşteri 2017'in 8.ayında aldığı 9_4, 46_4 hizmetleri bir sepeti;
# 2017’in 10.ayında aldığı  9_4, 38_4  hizmetleri başka bir sepeti ifade etmektedir.
# Sepetleri unique bir ID ile tanımlanması gerekmektedir.
# Bunun için öncelikle sadece yıl ve ay içeren yeni bir date değişkeni oluşturunuz.
# UserID ve yeni oluşturduğunuz date değişkenini "_"
# ile birleştirirek ID adında yeni bir değişkene atayınız.

df.info()
df.CreateDate = pd.to_datetime(df["CreateDate"])
df["New_CreateDate_YM"] = df["CreateDate"].dt.year.astype(str) + "_" + df['CreateDate'].dt.month.astype(str)
df["New_CreateDate_YM"].head()
df.head()

df["SepetId"] = df["UserId"].astype(str) + "_" + df["New_CreateDate_YM"].astype(str)

df.head()

#########################
# GÖREV 2: Birliktelik Kuralları Üretiniz
#########################

# Adım 1: Aşağıdaki gibi sepet hizmet pivot table’i oluşturunuz.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

df.groupby(["SepetId", "Hizmet"])["CategoryId"].count().unstack().fillna(0).applymap(lambda x: 1 if x>0 else 0).head()

df_new = df.groupby(["SepetId", "Hizmet"])["CategoryId"].count().unstack().fillna(0).applymap(lambda x: 1 if x>0 else 0)
df_new.shape
df_new.head()

# Adım 2: Birliktelik kurallarını oluşturunuz.

frequent_itemsets = apriori(df_new, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head()
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.shape
rules.head()

sorted_rules = rules.sort_values("confidence", ascending=False)
sorted_rules.head()

#Adım 3: arl_recommender fonksiyonunu kullanarak en son 2_0 hizmetini alan bir kullanıcıya hizmet önerisinde bulununuz.

recommendation_list = []

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("confidence", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list

arl_recommender(rules, "2_0")
['15_1', '22_0', '25_0', '13_11', '38_4']
