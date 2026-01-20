import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('rounded_hours_student_scores.csv', encoding='latin1')
df.head()

df.nunique()

x = df['Hours']
y = df['Scores']
plt.scatter(x, y, color='red')
plt.title("Hours vs Scores")
plt.xlabel("Hours studied")
plt.ylabel("Score secured")
plt.show()


x_train, x_test, y_train, y_test = train_test_split(x.values.reshape(-1,1), y, test_size= 0.2, random_state=42 ) 


ln_model = LinearRegression()
ln_model.fit(x_train, y_train)

print("Intercept(b0):", ln_model.intercept_)
print("slope(b1):", ln_model.coef_)

y_pred = ln_model.predict(x_test)

# Plot training data
plt.scatter(x_train, y_train, color='red')

# Regression line over entire range (training + testing)
plt.plot(x_train, ln_model.predict(x_train), color='blue', label="Regression Line")
plt.title("Hours vs Scores with Regression Line")
plt.xlabel("Hours studied")
plt.ylabel("Score secured")
plt.legend()
plt.show()


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R²:", r2)

# Save the trained model
joblib.dump(ln_model, 'student_score_model.pkl')
print("Model trained and saved successfully!")