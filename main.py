from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from sklearn.linear_model import LogisticRegression
from kivy.graphics import Color, Rectangle
import pandas as pd

# Example dataset
data = pd.DataFrame({
    'Age': [22, 25, 19, 30, 24, 27, 23, 21, 26, 28, 20, 22, 29, 25, 31, 24, 26, 21, 23, 27],
    'Training_Hours': [6, 5, 7, 4, 5, 6, 4, 8, 7, 5, 6, 7, 3, 5, 5, 7, 6, 6, 4, 5],
    'Previous_Injuries': [1, 0, 2, 1, 3, 0, 1, 2, 1, 4, 0, 1, 3, 1, 0, 2, 1, 0, 1, 2],
    'Fitness_Level': [8, 7, 9, 6, 8, 7, 7, 9, 8, 5, 7, 8, 6, 7, 9, 8, 7, 8, 7, 7],
    'Injury_Risk': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1]
})

class InjuryRiskApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Set the background color
        with layout.canvas.before:
            Color(0.95, 0.95, 0.95, 1)  # Light grey color
            self.rect = Rectangle(size=layout.size, pos=layout.pos)

        # Update the background with window size
        layout.bind(size=self._update_rect, pos=self._update_rect)

        # Add widgets for data input
        self.inputs = {}
        self.add_input_field(layout, 'Age:')
        self.add_input_field(layout, 'Training Hours:')
        self.add_input_field(layout, 'Previous Injuries (Count):')
        self.add_input_field(layout, 'Fitness Level (1-10):')

        # Button to assess risk
        assess_button = Button(text="Assess Injury Risk", on_press=self.assess_risk)
        layout.add_widget(assess_button)

        # Label to show the result
        self.result_label = Label(text="", color=(1, 0, 0, 1))  # Red text color for results
        layout.add_widget(self.result_label)

        # Prepare the logistic regression model
        self.prepare_model()

        return layout

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def add_input_field(self, layout, label_text):
        box = BoxLayout(orientation='horizontal', spacing=10)
        label = Label(text=label_text, size_hint_x=0.3, color=(0, 0, 0, 1))  # Black text color
        text_input = TextInput(multiline=False, size_hint_x=0.7,
                               foreground_color=(0, 0, 0, 1))  # Black text color for input
        box.add_widget(label)
        box.add_widget(text_input)
        layout.add_widget(box)
        self.inputs[label_text] = text_input

    def prepare_model(self):
        X = data[['Age', 'Training_Hours', 'Previous_Injuries', 'Fitness_Level']]
        y = data['Injury_Risk']
        self.model = LogisticRegression()
        self.model.fit(X, y)

    def assess_risk(self, instance):
        try:
            age = float(self.inputs['Age:'].text)
            training_hours = float(self.inputs['Training Hours:'].text)
            previous_injuries = float(self.inputs['Previous Injuries (Count):'].text)
            fitness_level = float(self.inputs['Fitness Level (1-10):'].text)

            # Predict injury risk
            prediction = self.model.predict([[age, training_hours, previous_injuries, fitness_level]])[0]
            risk_prediction = "High Risk" if prediction == 1 else "Low Risk"
            self.result_label.text = f"Injury Risk: {risk_prediction}"
        except ValueError:
            self.result_label.text = "Invalid input. Please enter numerical values."

if __name__ == '__main__':
    InjuryRiskApp().run()
