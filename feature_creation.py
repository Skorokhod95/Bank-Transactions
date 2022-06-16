class feature_creation():
    def __init__(self, expert_type=False, expert_code=False):
        
        import holidays 
        import pandas as pd
        import numpy as np
        import os
        import datetime

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from catboost import CatBoostClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import BaggingClassifier

        from sklearn.model_selection import KFold
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, auc, roc_curve

        import gensim.downloader as api
        from multiprocessing import cpu_count
        from gensim.models.word2vec import Word2Vec
        from sklearn.feature_extraction.text import TfidfTransformer

        from statsmodels.stats.contingency_tables import mcnemar
        from scipy.stats import ttest_ind
        
        if expert_type:
            self.expert_type = expert_type
        else:
            self.expert_type = [1010, 1110, 1100, 1200]
            
        if expert_code:
            self.expert_code = expert_code
        else:
            self.expert_code = [6011, 6010, 4814, 5411,
                                4829, 5499, 5912, 5541,
                                5331, 5814, 5999, 5921, 5311]
        
    def get_codes_types(self, X, types, codes):
        self.type = types.merge(X, how='outer', left_on='type', right_on='type').type.drop_duplicates()
        self.code = codes.merge(X, how='outer', left_on='code', right_on='code').code.drop_duplicates()

    
    def tf_idf(self, X, t, c):
        transformer = TfidfTransformer(smooth_idf=False)
        
        
        dataset_t = pd.DataFrame(t, columns=['type']).merge(X[['type', 'client_id', 'code']], how='left', left_on='type', right_on='type')\
                                           .fillna('client_0')\
                                           .pivot_table(columns='type', index='client_id', aggfunc='count').fillna(0)\
                                           .drop(index='client_0', errors='ignore')
        #dataset_t = X[X.type.isin(self.type)].pivot_table(values = 'code', index='client_id', columns='type', aggfunc = 'count')\
        #                                     .fillna(0)
        
        tfidf = transformer.fit_transform(dataset_t.values)
        type_ = pd.DataFrame(tfidf.toarray(), columns=['type_'+str(i) for i in dataset_t.columns])
        
        #dataset_c = X[X.code.isin(self.code)].pivot_table(values='type', index='client_id', columns='code', aggfunc = 'count')\
        #                                     .fillna(0)
        dataset_c = pd.DataFrame(c, columns=['code']).merge(X[['type', 'client_id', 'code']], how='left', left_on='code', right_on='code')\
                                           .fillna('client_0')\
                                           .pivot_table(columns='code', index='client_id', aggfunc='count').fillna(0)\
                                           .drop(index='client_0', errors='ignore')
        tfidf = transformer.fit_transform(dataset_c.values)
        code_ = pd.DataFrame(tfidf.toarray(), columns=['code_'+str(i) for i in dataset_c.columns])
        
        return pd.concat([code_, type_, pd.DataFrame(dataset_c.index)], axis=1)
    
    def seasons(x):
        season = {
            1:'winter',
            2:'winter',
            3:'spring',
            4:'spring',
            5:'spring',
            6:'summer',
            7:'summer',
            8:'summer',
            9:'autumn',
            10:'autumn',
            11:'autumn',
            12:'winter'
        }
        return season[x.month]
    
    def add_time_features(self, X):
        X['date'] = X.datetime.apply(lambda x: x.split(' ')[0]).astype(int)
        X['time1'] = X.datetime.apply(lambda x: x.split(' ')[1].split(':')[0]).astype(int)
        X['time2'] = X.datetime.apply(lambda x: x.split(' ')[1].split(':')[1]).astype(int)
        X['time3'] = X.datetime.apply(lambda x: x.split(' ')[1].split(':')[2]).astype(int)
        X.time2[X.time3>=60] = X.time2[X.time3>=60]+1
        X.time3[X.time3>=60] = 0

        X.time1[X.time2>=60] = X.time1[X.time2>=60]+1
        X.time2[X.time2>=60] = 0

        X.date[X.time1>=24] = X.date[X.time1>=24]+1
        X.time1[X.time1>=24] = 0
        
        X['time'] = X[['time1', 'time2', 'time3']].T.apply(lambda x: datetime.time(*x))
        X['date'] = pd.to_datetime('20160101') + X.date.apply(lambda x: pd.Timedelta(x, 'd'))
        X.datetime = X[['date', 'time']].T.apply(lambda x: datetime.datetime.combine(*x))
        
        X['weekday'] = X.date.apply(lambda x: x.weekday())
        X['weekend'] = X.date.apply(pd.Timestamp.date).isin(holidays.Russia(years = 2017).keys()) | \
                       X.date.apply(pd.Timestamp.date).isin(holidays.Russia(years = 2016).keys()) | \
                       X.weekday.isin([5,6])
        X['season'] = X.date.apply(seasons)
        X['month'] = X.date.apply(lambda x: x.month)
        X['year'] = X.date.apply(lambda x: x.year)
        
        a = X[['date', 'weekday']].drop_duplicates().sort_values(by='date')
        week = (((a.date.iloc[0] - a.date).abs() + pd.Timedelta(days = a.iloc[0].weekday)) / 7).apply(lambda x: x.days)
        week = pd.DataFrame([week.values, a.date.values]).T
        X['week'] = X.date.apply(lambda x: week[0][x == week[1]].iloc[0])
        
        return X

        
    def seasonality(self, X):
        c = X['sum']<0
        weekday_fcounts = X.groupby(by=['date', 'client_id', 'weekday'])\
                           .count()\
                           .time1\
                           .groupby(by=['client_id', 'weekday'])\
                           .mean()\
                           .reset_index()\
                           .pivot_table(index='client_id', columns='weekday')\
                           .fillna(0)
        weekday_fexpenses = X[c].groupby(by=['date', 'client_id', 'weekday'])\
                                .sum()\
                                ['sum']\
                                .groupby(by=['client_id', 'weekday'])\
                                .apply(lambda x: x.median())\
                                .reset_index()\
                                .pivot_table(index='client_id', columns='weekday')\
                                .fillna(0)
        weekday_freceipts = X[~c].groupby(by=['date', 'client_id', 'weekday'])\
                                 .sum()\
                                 ['sum']\
                                 .groupby(by=['client_id', 'weekday'])\
                                 .apply(lambda x: x.median())\
                                 .reset_index()\
                                 .pivot_table(index='client_id', columns='weekday')\
                                 .fillna(0)
        weekday_f = weekday_fcounts.merge(weekday_fexpenses, how='outer', left_on='client_id', right_on='client_id')\
                                   .merge(weekday_freceipts, how='outer', left_on='client_id', right_on='client_id')\
                                   .fillna(0)

        weekend_fexpenses = X[c].groupby(by=['client_id', 'week', 'weekend'])\
                            .sum()\
                            ['sum']\
                            .groupby(by=['client_id', 'weekend'])\
                            .mean()\
                            .reset_index()\
                            .pivot_table(columns='weekend', index='client_id')\
                            .fillna(0)
        weekend_freceipts = X[~c].groupby(by=['client_id', 'week', 'weekend'])\
                                    .sum()\
                                    ['sum']\
                                    .groupby(by=['client_id', 'weekend'])\
                                    .mean()\
                                    .reset_index()\
                                    .pivot_table(columns='weekend', index='client_id')\
                                    .fillna(0)
        weekend_fcounts = X.groupby(by=['client_id', 'week', 'weekend'])\
                                .count()\
                                ['sum']\
                                .groupby(by=['client_id', 'weekend'])\
                                .mean()\
                                .reset_index()\
                                .pivot_table(columns='weekend', index='client_id')\
                                .fillna(0)
        weekend_f = weekend_fcounts.merge(weekend_fexpenses, how='outer', left_on='client_id', right_on='client_id')\
                                    .merge(weekend_freceipts, how='outer', left_on='client_id', right_on='client_id')\
                                    .fillna(0)
        
        season_fexpenses = X[c].groupby(by=['client_id', 'year', 'season'])\
                                .sum()\
                                ['sum']\
                                .groupby(by=['client_id', 'season'])\
                                .mean()\
                                .reset_index()\
                                .pivot_table(columns='season', index='client_id')\
                                .fillna(0)

        season_freceipts = X[~c].groupby(by=['client_id', 'year', 'season'])\
                                    .sum()\
                                    ['sum']\
                                    .groupby(by=['client_id', 'season'])\
                                    .mean()\
                                    .reset_index()\
                                    .pivot_table(columns='season', index='client_id')\
                                    .fillna(0)
        season_fcounts = X.groupby(by=['client_id', 'year', 'season'])\
                                .count()\
                                ['sum']\
                                .groupby(by=['client_id', 'season'])\
                                .mean()\
                                .reset_index()\
                                .pivot_table(columns='season', index='client_id')\
                                .fillna(0)
        season_f = season_fcounts.merge(season_fexpenses, how='outer', left_on='client_id', right_on='client_id')\
                                    .merge(season_freceipts, how='outer', left_on='client_id', right_on='client_id')\
                                    .fillna(0)
        
        weekend_f.columns = [
            'count_ww', 'count_w',
            'expenses_ww', 'expenses_w',
            'receipts_ww', 'receipts_w'
        ]
        season_f.columns = [
            'autumn_counts', 'spring_counts', 'summer_counts', 'winter_counts',
            'autumn_expenses', 'spring_expenses', 'summer_expenses', 'winter_expenses',
            'autumn_receipts', 'spring_receipts', 'summer_receipts', 'winter_receipts'
        ]
        weekday_f.columns = [
            '0_counts', '1_counts', '2_counts', '3_counts', '4_counts', '5_counts', '6_counts',
            '0_expenses', '1_expenses', '2_expenses', '3_expenses', '4_expenses', '5_expenses', '6_expenses',
            '0_receipts', '1_receipts', '2_receipts', '3_receipts', '4_receipts', '5_receipts', '6_receipts'
        ]
        return weekend_f.merge(season_f, how='outer', left_on='client_id', right_on='client_id')\
                          .merge(weekday_f, how='outer', left_on=['client_id'], right_on=['client_id'])\
                          .fillna(0)\
                          .reset_index()

    def expert(self, X):
        new_code_type = X[X.type.isin(self.expert_type) & ~X.code.isin(self.expert_code)]
        return self.tf_idf(new_code_type, self.expert_type, self.code[~self.code.isin(self.expert_code)])
        
    def client_description(self, X):
        #Максимальная трата клиента
        expenses_max = X[X['sum']<0].groupby(by='client_id')['sum'].min()
        #Минимальная трата клиента
        expenses_min = X[X['sum']<0].groupby(by='client_id')['sum'].max()
        sum_ = expenses_max.to_frame().merge(expenses_min, how='outer', left_on='client_id', right_on='client_id')
        #Срадняя трата клиента
        expenses_mean = X[X['sum']<0].groupby(by='client_id')['sum'].mean()
        sum_ = sum_.merge(expenses_mean, how='outer', left_on='client_id', right_on='client_id')
        #Максимальное поступление клиента
        receipt_max = X[X['sum']>0].groupby(by='client_id')['sum'].max()
        sum_ = sum_.merge(receipt_max, how='outer', left_on='client_id', right_on='client_id')
        #Минимаотное поступление клиента
        receipt_min = X[X['sum']>0].groupby(by='client_id')['sum'].min()
        sum_ = sum_.merge(receipt_min, how='outer', left_on='client_id', right_on='client_id')
        #Среднее поступление клиента
        receipt_mean = X[X['sum']>0].groupby(by='client_id')['sum'].mean()
        sum_ = sum_.merge(receipt_mean, how='outer', left_on='client_id', right_on='client_id')
        sum_.fillna(0, inplace=True)
        sum_.columns = ['expenses_max', 'expenses_min', 'expenses_mean', 'receipt_max', 'receipt_min', 'receipt_mean']
        return sum_

    
    def get_labels(self, X, y):
        return pd.concat([X,y])[['client_id', 0]].groupby(by='client_id').mean()
    
    def get_data(self,  X, types, codes, datatype='train'):
        if datatype=='test':
            try:
                self.code
            except Exception:
                print("Haven't been transformed train data yet. Transform train data first, then test data.")
                return None
        else:
            self.get_codes_types(X, types, codes)   
        out  = self.tf_idf(X, self.type, self.code).merge(self.expert(X), how = 'outer', left_on='client_id', right_on = 'client_id')\
                             .merge(self.seasonality(self.add_time_features(X)), how = 'outer', left_on='client_id', right_on = 'client_id')\
                             .merge(self.client_description(X), how = 'outer', left_on='client_id', right_on = 'client_id')\
                             .fillna(0)
        return out
    