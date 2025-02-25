import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot

#%%
data = pd.read_csv('cars.csv')
data.shape
#%%
#Cleanning database
data.drop(['Unnamed: 0'],axis=1,inplace=True)
data.dropna()
data.head()
#%%
#Velocidade x  distancia de parada
x=data.iloc[:,1].values
y=data.iloc[:,0].values

#%%
#Matriz de correlacao das variaveis
correlation = np.corrcoef(x,y)
#%%
#Fit do modelo de regressao linear simples
x=x.reshape(-1,1)
model = LinearRegression()
model.fit(x,y)
#%%
#Intercepto e Coeficiente angular
intercept = model.intercept_
coef = model.coef_
#%%
#Plot do gráfico da regressao
plt.scatter(x,y)
plt.plot(x,model.predict(x),color='red')
#%%
#Plot da distribuicao dos residuos
residuals = ResidualsPlot(model)
residuals.fit(x,y)
residuals.poof
#%%
#Usando pacote statsmodel para estimacao de modelos
import statsmodels.api as sm
model_sm = sm.OLS.from_formula('speed~dist',data).fit()
model_sm.summary()

model_sm.conf_int(alpha=0.05)
#%%
#Plotando Um novo gráfico com os Intervalos de Confinaca
plt.figure(figsize=(15,10))
sns.regplot(data=data, x='dist', y='speed', marker='o', ci=95,
            scatter_kws={"color":'navy', 'alpha':0.7, 's':220},
            line_kws={"color":'grey', 'linewidth': 5})
plt.title('CI: 95%', fontsize=30)
plt.xlabel('Distânce', fontsize=24)
plt.ylabel('Speed', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0, 35)
plt.ylim(0, 60)
plt.legend(['True Values', 'Fitted Values', '95% CI'],
           fontsize=24, loc='upper left')
plt.show()

#%%
# Teste de aderencia aos resíduos a distribuicao Normal

# Teste de Shapiro-Wilk (n < 30)
# from scipy.stats import shapiro
# shapiro(model_sm.resid)

# Teste de Shapiro-Francia (n >= 30)
# Carregamento da função 'shapiro_francia' do pacote 'statstests.tests'

from statstests.tests import shapiro_francia

teste_sf = shapiro_francia(model_sm.resid)
teste_sf= teste_sf.items()
method, statistics_W, statistics_z, p = teste_sf
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
    print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')
#%%
# Histograma dos resíduos do modelo OLS linear

plt.figure(figsize=(15,10))
hist1 = sns.histplot(data=model_sm.resid, kde=True, bins=25,
                     color = 'darkorange', alpha=0.4, edgecolor='silver',
                     line_kws={'linewidth': 3})
hist1.get_lines()[0].set_color('orangered')
plt.xlabel('Resíduals', fontsize=20)
plt.ylabel('Frequence', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()

#%%
# Como temos residuos aderentes a normalidade e os parametros estatisticamente significativos
# Vemos que a regressao linear eh adequada para modelar esses dados
# Vamos complementar com a transformacao de Box-Cox para ver como a regressao se comporta

from scipy.stats import boxcox

yast, lmbda = boxcox(data['speed'])
data['bc_speed'] = yast

model_bc = sm.OLS.from_formula('bc_speed~dist', data).fit()
model_bc.summary()
model_bc.conf_int(alpha=0.05)
#%%
# Teste de Shapiro-Francia para modelo de Box-Cox
teste_sf = shapiro_francia(model_bc.resid)
teste_sf=teste_sf.items()
method, statistics_W, statistics_z, p = teste_sf
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
    print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')
#%%

# Plot do Histograma dos residuos do modelo com tranf Box-Cox

plt.figure(figsize=(15,10))
hist1 = sns.histplot(data=model_bc.resid, kde=True, bins=25,
                     color = 'darkorange', alpha=0.4, edgecolor='silver',
                     line_kws={'linewidth': 3})
hist1.get_lines()[0].set_color('orangered')
plt.xlabel('Resíduals', fontsize=20)
plt.ylabel('Frequence', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()
#%%
# Revertendo a transformacao e comparando os modelos
data['model_linear'] = model_sm.fittedvalues
data['model_bc'] = (model_bc.fittedvalues*lmbda+1)**(1/lmbda)

# Plot dos modelos; pode-se confirmar que temos boa precisao ja com o modelo linear

plt.figure(figsize=(15,10))
sns.scatterplot(x="dist", y="speed", data=data, color='grey',
                s=350, label='Valores Reais', alpha=0.7)
sns.regplot(x="dist", y="model_bc", data=data, order=lmbda,
            color='darkviolet', ci=False, scatter=False, label='Box-Cox',
            line_kws={'linewidth': 2.5})
sns.scatterplot(x="dist", y="model_bc", data=data, color='darkviolet',
                s=200, label='Fitted Values Box-Cox', alpha=0.5)
sns.regplot(x="dist", y="model_linear", data=data,
            color='darkorange', ci=False, scatter=False, label='OLS Linear',
            line_kws={'linewidth': 2.5})
sns.scatterplot(x="dist", y="model_linear", data=data, color='darkorange',
                s=200, label='Fitted Values OLS Linear', alpha=0.5)
plt.title('Dispersion of data and adjustments of OLS linear and Box-Cox models',
          fontsize=20)
plt.xlabel('Distânce until stop', fontsize=17)
plt.ylabel('Speed', fontsize=17)
plt.legend(loc='lower right', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
#%%
#############################################################################
#                        REGRESSÃO NÃO LINEAR MÚLTIPLA                      #
#############################################################################

database = pd.read_csv('mt_cars.csv')
database.shape

# Clean database
cars = database['Unnamed: 0']
database.drop(['Unnamed: 0'],axis=1,inplace=True)

x=database.iloc[:,1:]
y=database.iloc[:,0]
#%%
# Descriptive analysis and check of correlations between variables
database.describe()

correlation_matrix = x.corr() #Some variables have high correlation
plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f",
                      cmap=plt.cm.viridis_r,
                      annot_kws={'size': 25}, vmin=-1, vmax=1)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=15)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=15)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=17)
plt.show()

import pingouin as pg

correlation_matrix2 = pg.rcorr(x, method='pearson',
                              upper='pval', decimals=4,
                              pval_stars={0.01: '***',
                                          0.05: '**',
                                          0.10: '*'})

#%%
# First test of Multiple Linear Regression without Stepwise Procedure

model = sm.OLS.from_formula('mpg ~ cyl + disp + hp + drat + wt +\
                            qsec + vs + am + gear + carb', database).fit()
                            
model.summary() # Many variables do not pass on T-test 

teste_sf = shapiro_francia(model.resid) # Still, errors are adherent to Normal dist
teste_sf= teste_sf.items()
method, statistics_W, statistics_z, p = teste_sf
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 
if p[1] > alpha:
    print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')
#%%

#Now we use the stepwise procedure to produce a more reliable model

from statstests.process import stepwise

model_step = stepwise(model,pvalue_limit=0.05)

model_step.summary()

teste_sf = shapiro_francia(model.resid) 
teste_sf= teste_sf.items()
method, statistics_W, statistics_z, p = teste_sf
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 
if p[1] > alpha:
    print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')
#%%

#Finally, we use the Box-Cox transformation to obtain a final model

yast, lmbda = boxcox(y)

print("lambda: ",lmbda)

database['mpg_bc'] = yast

model_bc = sm.OLS.from_formula('mpg_bc ~ cyl + disp + hp + drat + wt +\
                            qsec + vs + am + gear + carb', database).fit()

model_bc.summary()

teste_sf = shapiro_francia(model_bc.resid) 
teste_sf= teste_sf.items()
method, statistics_W, statistics_z, p = teste_sf
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 
if p[1] > alpha:
    print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
 	print('Rejeita-se H0 - Distribuição não aderente à normalidade')
#%%

# Lastly, we test the Box-Cox transform with stepwise procedure

model_step_bc = stepwise(model_bc,pvalue_limit=0.05)

model_step_bc.summary()

teste_sf = shapiro_francia(model_step_bc.resid) 
teste_sf= teste_sf.items()
method, statistics_W, statistics_z, p = teste_sf
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 
if p[1] > alpha:
    print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
 	print('Rejeita-se H0 - Distribuição não aderente à normalidade')

# Recover the untransformed values
database['mpg_step_bc_final'] = (model_step_bc.fittedvalues*lmbda+1)**(1/lmbda)