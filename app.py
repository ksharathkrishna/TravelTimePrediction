from flask import Flask, render_template, flash, redirect, url_for, session, request, logging
from wtforms import Form, StringField, TextAreaField, PasswordField, validators, IntegerField


app = Flask(__name__)


@app.route('/')
def index():

    return render_template('home.html')


class RegisterForm(Form):
    source = IntegerField('source',[validators.Length(min=1, max=25)])
    destination = IntegerField('destination', [validators.Length(min=1, max=25)])
    timezone = IntegerField('timezone', [validators.Length(min=1, max=25)])
    day= IntegerField('day', [validators.Length(min=1, max=25)])
    weather = IntegerField('weather', [validators.Length(min=1, max=50)])
    temp = IntegerField('temp', [validators.Length(min=1, max=25)])



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = RegisterForm(request.form)

    if request.method == 'POST':
        src = form.source.data
        dst = form.destination.data
        tz=form.timezone.data
        d=form.day.data
        wtr = form.weather.data
        temp = form.temp.data
        import knn_tensor
        return render_template('result.html', src=src,dst=dst,a=knn_tensor.pred(src,dst,tz,d,wtr,temp))

if __name__ == '__main__':
    TEMPLATES_AUTO_RELOAD = True

    app.run(debug=True)

