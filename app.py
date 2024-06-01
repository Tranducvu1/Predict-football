from flask import Flask, render_template, request,jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)
# Function to transform points (log1p used here, adjust if needed)
def transform_points(points):
    return np.log1p(points)

def inverse_transform_points(transformed_points):
    return np.expm1(transformed_points)

drop_columns = ['competition_type','club_id', 'season', 'goals', 'date', 'game_id', 'yellow_cards', 'red_cards']
data_main_df = pd.read_csv('data.csv')  
data_main_df = data_main_df.drop(columns=drop_columns, axis=1)

def train_model(df, relevant_features, target_feature):
    df = df.reset_index(drop=True)
    X = df[relevant_features]
    y = transform_points(df[target_feature])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Encode categorical features
    label_encoder = LabelEncoder()
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = label_encoder.fit_transform(X_train[col])
        X_test[col] = label_encoder.transform(X_test[col])

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    return model, label_encoder, scaler

def get_home_advantage(hosting):
    return 1 if hosting == "Home" else 0
relevant_features = ['wins', 'losses', 'draw', 'goal_difference', 'home_club_formation', 'home_wins', 'away_wins']
target_feature = 'points'

# Include home advantage as a feature
df_home = data_main_df.copy()
df_home["home_advantage"] = df_home["hosting"].apply(get_home_advantage)
relevant_features.append("home_advantage")

model, label_encoder, scaler = train_model(df_home, relevant_features, target_feature)

def predict(home_team, away_team, hosting, df_home, model, label_encoder, scaler, relevant_features):


    home_team_data = df_home[df_home['club_name'] == home_team]
    away_team_data = df_home[df_home['club_name'] == away_team]

    if home_team_data.empty or away_team_data.empty:
        return None

    home_data = home_team_data[relevant_features].iloc[0].values
    away_data = away_team_data[relevant_features].iloc[0].values

    data = np.array([home_data, away_data])

    for i, col in enumerate(relevant_features):
        if df_home[col].dtype == 'object':
            data[:, i] = label_encoder.transform(data[:, i].astype(str))

    data_scaled = scaler.transform(data)

    y_pred_home = model.predict([data_scaled[0]])
    y_pred_away = model.predict([data_scaled[1]])

    predicted_home_points = np.round(inverse_transform_points(y_pred_home)[0] /20)
    predicted_away_points = np.round(inverse_transform_points(y_pred_away)[0] /20)

    result = pd.DataFrame({
    'Đội': [home_team, away_team],
    'Điểm dự đoán': [predicted_home_points, predicted_away_points]
    })

    result['Người chiến thắng là'] = ['Đội nhà' if predicted_home_points > predicted_away_points
                                  else 'Đội khách' if predicted_home_points < predicted_away_points
                                  else 'Hòa'
                                  for _ in range(len(result))]
    return result

laliga_df = data_main_df[data_main_df['competition_id'] == 'ES1']
es1_clubs_df = data_main_df[data_main_df['competition_id'] == 'L1']
nha_df = data_main_df[data_main_df['competition_id'] == 'GB1']
teams1 = es1_clubs_df['club_name'].unique()
teams2 = laliga_df['club_name'].unique()
teams3 = nha_df['club_name'].unique()

@app.route('/')
def home():
    return render_template('index.html', teams1=teams1)

@app.route('/page1')
def home2():
    return render_template('index2.html', teams2=teams2)
@app.route('/page2')
def home3():
    return render_template('index3.html', teams3=teams3)
@app.route('/predict', methods=['POST'])
def result():
    home_team = request.form['home_team']
    away_team = request.form['away_team']
    hosting = request.form['hosting']

    result = predict(home_team, away_team, hosting, df_home, model, label_encoder, scaler, relevant_features)
    if result is not None:
        return render_template('result.html', result=result.to_html())
    else:
        return "Không tìm thấy dữ liệu cho một hoặc cả hai đội."

if __name__ == '__main__':
    app.run(port=3000, debug=True)
