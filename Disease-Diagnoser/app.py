from flask import Flask,render_template,request
import predictions1,predictions2,predictions3,ANNheart,ANNmaleria,ANNdiabetes

app=Flask(__name__)

@app.route('/' ,methods=["GET","POST"])
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/pneumonia',methods=["GET","POST"])
def lungcancer1():
    return render_template("pneumonia.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/team')
def team():
    return render_template("team.html")

@app.route('/selection1',methods=["GET","POST"])
def selection1():
    return render_template("selection1.html")

@app.route('/selection2',methods=["GET","POST"])
def selection2():
    return render_template("selection2.html")

@app.route('/heartdisease1', methods=["GET","POST"])
def action1():
    return render_template("heartdisease1.html")

@app.route('/tuberculosis1', methods=["GET","POST"])
def action2():
    return render_template("tuberculosis1.html")



@app.route('/sub1',methods=["GET","POST"])
def sub1():
    if request.method=="POST":
        image=request.files['imagefile']
        image_path="static/"+image.filename
        image.save(image_path)
        pred=predictions1.prediction(path=image_path)
    print(pred)
    return render_template("sub1.html",img=image_path,n=pred)

@app.route('/sub2', methods=["GET", "POST"])
def sub2():
        if request.method == "POST":
            image = request.files['imagefile']
            image_path = "static/" + image.filename
            image.save(image_path)
            pred = predictions2.prediction(path=image_path)

        return render_template('sub2.html',img=image_path,n=pred)

@app.route('/sub3', methods=["GET", "POST"])
def sub3():
        if request.method == "POST":
            image = request.files['imagefile']
            image_path = "static/" + image.filename
            image.save(image_path)
            pred = predictions3.prediction(path=image_path)

        return render_template('sub3.html',img=image_path,n=pred)

@app.route('/diabetes',methods=["GET","POST"])
def diabetes():
        return render_template("diabetes.html")


@app.route('/sub4',methods=["GET","POST"])
def sub4():
    if request.method=="POST":

        userInput1=request.form.get("preg")
        userInput2=request.form.get("glucose")
        userInput3 = request.form.get("bpressure")
        userInput4 = request.form.get("i")
        userInput5 = request.form.get("bmi")
        x=[userInput1,userInput2,userInput3,userInput4,userInput5]
        x=ANNmaleria.ANNdiabetes(x)
    return render_template("sub4.html",n=x)

@app.route('/malaria',methods=["GET","POST"])
def malaria():
    return render_template("malaria.html")


@app.route('/sub5',methods=["GET","POST"])
def sub5():
    if request.method=="POST":
        userInput1 =request.form.get("preg")
        userInput2 =request.form.get("glucose")
        userInput3 = request.form.get("bpressure")
        userInput4 = request.form.get("i")
        userInput5 = request.form.get("bmi")
        x=[userInput1,userInput5,userInput4,userInput3,userInput2]
        x=list(map(int,x))
        print(x)
        x = ANNmaleria.ANNdiabetes(x)
    return render_template("sub5.html",n=x)


@app.route('/heart stroke',methods=["GET","POST"])
def heartstroke():
    return render_template("heart stroke.html")

@app.route('/sub6',methods=["GET","POST"])
def sub6():
    if request.method=="POST":
        userInput1=request.form.get("preg")
        userInput2=request.form.get("glucose")
        userInput3 = request.form.get("bpressure")
        userInput4 = request.form.get("i")
        userInput5 = request.form.get("bmi")
        x=[userInput1,userInput5,userInput4,userInput3,userInput2]
        x=ANNmaleria.ANNdiabetes(x)
    return render_template("sub6.html",n=x)

if __name__=="__main__":
    app.run(debug=True)