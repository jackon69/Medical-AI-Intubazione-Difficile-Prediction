# 🔥 VERSIONE AUTOMATICA - ESEGUI QUESTA UNICA CELLA
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import joblib
import os

print("🚀 INIZIALIZZAZIONE SISTEMA MEDICAL AI...")

# 1. Classe Database
class MedicalDatabase:
    def __init__(self):
        self.train_file = 'train_dataset.csv'
        self.test_file = 'test_dataset.csv'
        self.new_data_file = 'new_cases.csv'
    
    def create_initial_datasets(self, n_samples=1000):
        np.random.seed(42)
        data = {
            'eta': np.clip(np.random.normal(55, 15, n_samples), 18, 90).astype(int),
            'peso': np.clip(np.random.normal(75, 20, n_samples), 40, 150),
            'mallampati': np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.35, 0.2, 0.05]),
            'stop_bang': np.random.choice(range(0, 9), n_samples, p=[0.1, 0.15, 0.2, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04]),
            'al_ganzuri': np.clip(np.random.normal(4.5, 1.2, n_samples), 2, 8),
            'dimensioni': np.clip(np.random.normal(16, 2, n_samples), 10, 25),
            'dii': np.clip(np.random.normal(5.5, 1.5, n_samples), 2, 10),
            'cormack': np.random.choice([1, 2, 3, 4], n_samples, p=[0.5, 0.3, 0.15, 0.05])
        }
        df = pd.DataFrame(data)
        df['intubazione_difficile'] = (df['cormack'] > 2).astype(int)
        train_df = df.iloc[:800].copy()
        test_df = df.iloc[800:].copy()
        train_df.to_csv(self.train_file, index=False)
        test_df.to_csv(self.test_file, index=False)
        pd.DataFrame(columns=train_df.columns).to_csv(self.new_data_file, index=False)
        print("✅ Database creato!")
        return train_df, test_df

# 2. Sistema AI
class ColabMedicalAI:
    def __init__(self):
        self.db = MedicalDatabase()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['eta', 'peso', 'mallampati', 'stop_bang', 'al_ganzuri', 'dimensioni', 'dii']
    
    def initialize_system(self):
        try:
            self.train_df = pd.read_csv('train_dataset.csv')
            self.test_df = pd.read_csv('test_dataset.csv')
            print("✅ Database caricati")
        except:
            print("📁 Creazione nuovo database...")
            self.train_df, self.test_df = self.db.create_initial_datasets()
        
        try:
            self.model = joblib.load('medical_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            print("✅ Modello pre-esistente caricato")
        except:
            print("🤖 Addestramento nuovo modello...")
            self.train_model()
    
    def train_model(self):
        X_train = self.train_df[self.feature_columns]
        y_train = self.train_df['intubazione_difficile']
        X_test = self.test_df[self.feature_columns]
        y_test = self.test_df['intubazione_difficile']
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = RandomForestClassifier(n_estimators=150, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Valutazione
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        joblib.dump(self.model, 'medical_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        print(f"📈 Modello addestrato - AUC: {auc_score:.4f}")
    
    def predict_case(self, features):
        input_array = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
        input_scaled = self.scaler.transform(input_array)
        probability = self.model.predict_proba(input_scaled)[0, 1]
        return probability

# 3. INIZIALIZZA E TESTA
print("🎯 INIZIALIZZAZIONE IN CORSO...")
ai_system = ColabMedicalAI()
ai_system.initialize_system()

# 4. TEST CON CASI ESEMPIO
print("\n" + "="*60)
print("🎯 TEST AUTOMATICO CON CASI ESEMPIO")
print("="*60)

casi_test = [
    {
        'eta': 65, 'peso': 85, 'mallampati': 3, 'stop_bang': 6,
        'al_ganzuri': 3.5, 'dimensioni': 18, 'dii': 4.5,
        'nome': "PAZIENTE ALTO RISCHIO"
    },
    {
        'eta': 45, 'peso': 70, 'mallampati': 1, 'stop_bang': 2,
        'al_ganzuri': 5.0, 'dimensioni': 15, 'dii': 6.0,
        'nome': "PAZIENTE BASSO RISCHIO"
    },
    {
        'eta': 55, 'peso': 80, 'mallampati': 2, 'stop_bang': 4,
        'al_ganzuri': 4.5, 'dimensioni': 16, 'dii': 5.5,
        'nome': "PAZIENTE RISCHIO MODERATO"
    }
]

for caso in casi_test:
    features = {k: v for k, v in caso.items() if k != 'nome'}
    probabilita = ai_system.predict_case(features)
    
    print(f"\n📋 {caso['nome']}:")
    print(f"   Età: {caso['eta']} | Peso: {caso['peso']} | Mallampati: {caso['mallampati']}")
    print(f"   STOP-BANG: {caso['stop_bang']} | Al-Ganzuri: {caso['al_ganzuri']}cm")
    print(f"   Dimensioni: {caso['dimensioni']}cm | DII: {caso['dii']}cm")
    print(f"   🎯 PROBABILITÀ INTUBAZIONE DIFFICILE: {probabilita:.1%}")
    
    if probabilita > 0.5:
        print("   ⚠️  RISCHIO: ALTO - Preparare strumentazione speciale")
    else:
        print("   ✅ RISCHIO: BASSO - Procedura standard")

print("\n" + "="*60)
print("🎉 SISTEMA PRONTO! Puoi ora usare le funzioni manualmente.")
print("Per predire un nuovo caso, usa:")
print("ai_system.predict_case({'eta': ..., 'peso': ..., ...})")
