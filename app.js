var http = require("http");
var path = require("path");
var express = require("express");
const KerasJS = require('keras-js')
var mnist = require('mnist')
var argmax = require('compute-argmax')
var bodyParser = require('body-parser')

var app = express();

app.set("views", path.resolve(__dirname, "views"));
app.set("view engine", "ejs");
//app.use(express.static(__dirname + '/views'));

const model = new KerasJS.Model({
	filepaths: {
		model: 'image2digit_keras_json',
		weights: 'image2digit_keras_weights.buf',
		metadata: 'image2digit_keras_metadata.json'
	},
	filesystem: true
})

var set = mnist.set(8000,2000);
var trainingSet = set.training;
var testSet = set.test;
data = testSet[3].input;
test_value = argmax(testSet[3].output);
console.log("Testing Value: " + test_value)



app.get("/", function(request, response) {
 response.render("index");
});

//Gotta Use ajax to pass the number from html form to node, and display result
//After that find a way to save image from runtime to another variable, reshape it to (None, 784) 
//pass the image value using ajax to server
//and display the result.

var urlencodedparser = bodyParser.urlencoded({extended:false})
app.post('/ajax', urlencodedparser, function (req, res){  
   console.log(req.body)
   //var x = document.createElement("div");
   //window.document.write(req.body.field2)

   model.ready()
	.then(() => {
		const inputData = {
			'input': new Float32Array(data)
		}
		return model.predict(inputData)
	})
	.then(outputData => {
		var max = -1, j = -1;
		for(var i = 0 ; i < 10 ; i++){
			if(outputData['output'][i] > max){
				max = outputData['output'][i];
				j = i;
			}
		}

		predicted_value = j;
	    console.log("Algorithm Prediction: " + predicted_value);
	})
	.catch(err => {
		console.log("Not gotcha");
	})


   res.redirect('/');

});


app.listen(3000, function() {
  console.log('MNIST keras app started at port 3000');
});