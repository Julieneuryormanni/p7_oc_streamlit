import pandas as pd
import streamlit as st
import pickle
import plotly_express as px
import numpy as np
import matplotlib.pyplot as plt

lgb = open("LightGBMModel.pkl","rb")
lgbm = pickle.load(lgb)
km = open("Kmean.pkl","rb")
kmean = pickle.load(km)
data_client = pd.read_feather('data_client')
data_client.sort_values(by=['SK_ID_CURR'], inplace = True)

#Préparation du df qui va servir à la prédiction
features = ['SK_ID_CURR','NEW_EXT_MEAN',
 'EXT_SOURCE_3',
 'OWN_CAR_AGE',
 'CODE_GENDER',
 'AMT_ANNUITY',
 'EXT_SOURCE_2',
 'PAYMENT_RATE',
 'EXT_SOURCE_1',
 'AMT_CREDIT',
 'ANNUITY_INCOME_PERC',
 'DAYS_EMPLOYED',
 'NEW_APP_EXT_SOURCES_PROD',
 'AMT_GOODS_PRICE',
 'NAME_EDUCATION_TYPE_HIGHER_EDUCATION',
 'DAYS_BIRTH',
 'NEW_GOODS_CREDIT',
 'AMT_INCOME_TOTAL',
 'PREV_CHANNEL_TYPE_CREDIT_AND_CASH_OFFICES_MEAN',
 'PREV_NAME_YIELD_GROUP_LOW_ACTION_MEAN',
 'DAYS_EMPLOYED_PERC',
 'INS_DBD_STD',
 'APPROVED_AMT_ANNUITY_MEDIAN',
 'CLOSED_DAYS_CREDIT_ENDDATE_STD',
 'OCCUPATION_TYPE_CORE_STAFF',
 'INS_AMT_PAYMENT_MEDIAN',
 'INS_DPD_MEAN',
 'AMT_REQ_CREDIT_BUREAU_QRT',
 'INS_DBD_SUM',
 'FLAG_OWN_CAR',
 'INS_AMT_PAYMENT_MIN',
 'BUREAU_DAYS_CREDIT_UPDATE_MAX',
 'REGION_POPULATION_RELATIVE',
 'BUREAU_AMT_CREDIT_SUM_MEAN',
 'DAYS_REGISTRATION',
 'DAYS_ID_PUBLISH',
 'APPROVED_AMT_ANNUITY_STD',
 'BUREAU_AMT_CREDIT_SUM_MEDIAN',
 'PREV_CHANNEL_TYPE_REGIONAL_LOCAL_MEAN',
 'INS_NUM_INSTALMENT_NUMBER_MAX',
 'NEW_C_GP',
 'INS_DBD_MEAN',
 'NAME_EDUCATION_TYPE_SECONDARY_SECONDARY_SPECIAL',
 'INS_AMT_PAYMENT_MAX',
 'INS_PAYMENT_DIFF_SUM',
 'INS_DPD_STD',
 'TOTALAREA_MODE',
 'NONLIVINGAPARTMENTS_AVG',
 'INCOME_PER_PERSON',
 'INS_DBD_MEDIAN',
 'APPROVED_HOUR_APPR_PROCESS_START_STD',
 'BUREAU_AMT_CREDIT_SUM_STD',
 'ACTIVE_DAYS_CREDIT_ENDDATE_MIN',
 'NAME_FAMILY_STATUS_MARRIED',
 'PREV_DAYS_LAST_DUE_MAX',
 'PREV_NAME_GOODS_CATEGORY_MOBILE_MEAN',
 'APPROVED_RATE_DOWN_PAYMENT_STD',
 'APPROVED_DAYS_DECISION_MEDIAN',
 'ACTIVE_DAYS_CREDIT_MAX',
 'DEF_30_CNT_SOCIAL_CIRCLE',
 'CLOSED_DAYS_CREDIT_MAX',
 'PREV_APP_CREDIT_PERC_STD',
 'PREV_HOUR_APPR_PROCESS_START_MEAN',
 'INS_DAYS_ENTRY_PAYMENT_MAX',
 'POS_MONTHS_BALANCE_MAX',
 'BUREAU_DAYS_CREDIT_UPDATE_MEDIAN',
 'NAME_INCOME_TYPE_WORKING',
 'CC_AMT_PAYMENT_TOTAL_CURRENT_MEAN',
 'CLOSED_DAYS_CREDIT_UPDATE_MAX',
 'POS_NAME_CONTRACT_STATUS_ACTIVE_MEAN',
 'INS_PAYMENT_DIFF_MAX',
 'CLOSED_DAYS_CREDIT_ENDDATE_MIN',
 'INS_NUM_INSTALMENT_NUMBER_SUM',
 'PREV_NAME_TYPE_SUITE_UNACCOMPANIED_MEAN',
 'PREV_DAYS_DECISION_MAX',
 'INS_DAYS_INSTALMENT_STD',
 'INS_AMT_INSTALMENT_MEDIAN',
 'PREV_DAYS_LAST_DUE_1ST_VERSION_MEAN',
 'PREV_DAYS_FIRST_DUE_MEDIAN',
 'INS_AMT_INSTALMENT_MEAN',
 'PREV_PRODUCT_COMBINATION_CASH_X_SELL_LOW_MEAN',
 'BUREAU_AMT_CREDIT_MAX_OVERDUE_STD',
 'POS_CNT_INSTALMENT_STD',
 'AMT_REQ_CREDIT_BUREAU_YEAR',
 'REFUSED_DAYS_DECISION_STD',
 'PREV_DAYS_LAST_DUE_1ST_VERSION_MAX',
 'INS_PAYMENT_DIFF_MEAN',
 'INCOME_PER_PERSON_PERC_PAYMENT_RATE',
 'INS_AMT_PAYMENT_STD',
 'DAYS_LAST_PHONE_CHANGE',
 'LIVINGAREA_MEDI',
 'INS_DBD_MAX',
 'PREV_NAME_PAYMENT_TYPE_CASH_THROUGH_THE_BANK_MEAN',
 'POS_CNT_INSTALMENT_FUTURE_STD',
 'PREV_APP_CREDIT_PERC_MEDIAN',
 'PREV_CNT_PAYMENT_MEAN',
 'PREV_CNT_PAYMENT_MEDIAN',
 'BUREAU_AMT_CREDIT_SUM_DEBT_MEDIAN',
 'POS_MONTHS_BALANCE_SIZE',
 'PREV_HOUR_APPR_PROCESS_START_MIN',
 'INS_AMT_INSTALMENT_MAX']
data_lgb = data_client[features]
data_lgb.set_index('SK_ID_CURR', inplace = True)

st.set_page_config(layout="wide")


# Titre de la page
st.title("Prêt à dépenser : Scoring client")
st.write("Cette application prédit le potentiel d'un client à rembourser un prêt")

#Fonction pour les graphes
def count_graph(df, x, color):
    group = pd.DataFrame(df.groupby([x,color])['SK_ID_CURR'].count()).reset_index()
    group.rename(columns = {'SK_ID_CURR':'count'}, inplace = True)
    succes = group[group[x] == 'Oui']
    succes['Pourcentage']= succes['count']/succes['count'].sum()*100
    Non_succes = group[group[x] == 'Non']
    Non_succes['Pourcentage']= Non_succes['count']/Non_succes['count'].sum()*100
    for_fig = pd.concat([succes, Non_succes])
    
    fig = px.bar(for_fig, x=x, y='Pourcentage', color=color)
    fig.update_xaxes(type='category', title_text = "Succès de remboursement")
    
    return for_fig, fig
# Fonctions pour les radars plot
def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1)
                     * (x2 - x1) + x1)
    return sdata

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=6):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,
                label = "axes{}".format(i))
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles,
                                         labels=variables, fontsize = 10, color = 'White')
        [txt.set_rotation(angle-90) for txt, angle
             in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
            
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i],
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2))
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i], fontsize = 8, color = 'grey')
            ax.spines["polar"].set_visible(True)
            ax.set_ylim(*ranges[i])
            ax.set_facecolor("black")
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

# Afficher les info clients sur le côté à partir de leur id    
id = st.sidebar.selectbox("Entrer l'ID du client", data_client['SK_ID_CURR'])
Good = """
<div style = "background-color: #33CC66; padding : 10px" >
<h3 style = "color:white; text-align:center;"> BON REMBOURSEUR</h3>
</div>
"""
Bad = """
<div style = "background-color: #FF4000; padding : 10px">
<h3 style = "color:white; text-align:center;"> MAUVAIS REMBOURSEUR</<h3>
</div>
"""
sample = data_lgb.loc[id].values.reshape(1,-1)

if st.sidebar.button("Prédiction"):
    if lgbm.predict(sample).item()==0:
        st.sidebar.markdown(Good, unsafe_allow_html=True)
    else:
        st.sidebar.markdown(Bad, unsafe_allow_html=True)
st.sidebar.header("Information du client")
st.sidebar.write('GENRE:',(data_client['Genre'].loc[data_client['SK_ID_CURR'] == id]).item())
st.sidebar.write('AGE:',(data_client['Age'].loc[data_client['SK_ID_CURR'] == id]).item(),' ans')
st.sidebar.write('CONTRAT ACTUEL:',(data_client['Contrat'].loc[data_client['SK_ID_CURR'] == id]).item())
st.sidebar.write('SITUATION FAMILIALE:',(data_client['Situation familiale'].loc[data_client['SK_ID_CURR'] == id]).item())
st.sidebar.write('EDUCATION:',(data_client['Education'].loc[data_client['SK_ID_CURR'] == id]).item())
salaire = (data_client['Salaire (K$)'].loc[data_client['SK_ID_CURR'] == id]).item()
st.sidebar.write('SALAIRE TOTAL :','{:,.0f}'.format(salaire), 'K$')
credit = (data_client['Crédit (K$)'].loc[data_client['SK_ID_CURR'] == id]).item()
st.sidebar.write('MONTANT DE CREDIT :','{:,.0f}'.format(credit), 'K$')
annuit = (data_client['Annuité(K$)'].loc[data_client['SK_ID_CURR'] == id]).item()
st.sidebar.write("MONTANT D'ANNUITE :",'{:,.0f}'.format(annuit), 'K$')

# Tableau profil similaire
simi_client = """
<p style = "color:grey; text-align:center; font-weight : bold; font-size : 25px"> Quelques clients similaires</p>
"""
st.markdown(simi_client, unsafe_allow_html=True)
col_std = [col for col in data_client.columns if '_std' in col]
X_kmean = data_client[col_std]
cluster = kmean.predict(X_kmean)
data_client['cluster'] = cluster
profile_cols = ['SK_ID_CURR','Genre', 'Contrat', 'Situation familiale', 'Education', "Début d'emploi (an)", 'cluster']
client_profile = data_client[profile_cols]
client_profile.set_index('SK_ID_CURR', inplace = True)
cluster = (data_client['cluster'].loc[data_client['SK_ID_CURR'] == id]).item()
simi = client_profile[client_profile['cluster']==cluster].drop('cluster', axis = 1).head()
st.table(simi)
    
# Séparer en deux colonnes
c1, c2 = st.columns((2, 2))

with c1:
# Radar plot 
    radar_client = """
    <p style = "color:grey; text-align:center; font-weight : bold; font-size : 25px"> Caractéristiques financières du client</p>
    """
    st.markdown(radar_client, unsafe_allow_html=True)
    target = lgbm.predict(data_lgb.values)
    data_client['TARGET']=target
    radar_cols = ['SK_ID_CURR', 'TARGET',
              'Salaire (K$)','Annuité(K$)', 'Crédit (K$)', 'Annuité/Revenu (%)', 'Age', 'cluster']
    data_radar = data_client[radar_cols]
    data_radar.set_index('SK_ID_CURR', inplace=True)
    variables = ['Salaire (K$)', 'Annuité(K$)', 'Crédit (K$)', 'Annuité/Revenu (%)',
       'Age']
    data_ex = data_radar[variables].loc[id].values
    ranges = [(30,1200 ),
         (0,80),
         (80, 3000),
         (0,100),
         (18,80)]  
    
    ok = data_radar[data_radar['TARGET'] == 0].groupby(['cluster']).mean()
    impayes = data_radar[data_radar['TARGET'] == 1].groupby(['cluster']).mean()
    # plotting
    fig1 = plt.figure(figsize=(5, 5), facecolor = 'black')
    radar = ComplexRadar(fig1, variables, ranges)
    radar.plot(data_ex, label = 'Mon client')
    radar.fill(data_ex, alpha=0.5)
    radar.plot(ok.drop('TARGET', axis=1).iloc[0],
               label='Moyenne des clients sans défaut de paiement',
               color='g')
    radar.plot(impayes.drop('TARGET', axis=1).iloc[0],
               label='Moyenne des clients avec défaut de paiement',
               color='r')
    legend_radar = fig1.legend(loc = 'lower left', fontsize = 8, facecolor = 'black')
    for t in legend_radar.get_texts():
        plt.setp(t, color = 'w')
    st.write(fig1)


with c2:
    #Graphes généralités
    general = """
    <p style = "color:grey; text-align:center; font-weight : bold; font-size : 25px"> Capacité de remboursement pour l'ensemble des clients de l'application (n=200)</p>
    """
    st.markdown(general, unsafe_allow_html=True)
    data_client["success"] = 'Oui'
    data_client["success"] = np.where((data_client["TARGET"].isin([1])), "Non", data_client["success"])
    variables = ['Genre', 'Situation familiale', 'Education']
    choix = st.radio("Choisir une variable", variables)
    for_fig, fig = count_graph(data_client, 'success', choix)
    st.write(fig) 
