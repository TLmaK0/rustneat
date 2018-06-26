var maxFitness = getParameterByName('max_fitness');
var fitnessAxis = {};
function fitness_init(id){
  var svg = d3.select('#' + id).append('svg');
  svg.append('path');

  fitnessAxis[id+"y"] = svg.append("g");
  fitnessAxis[id+"x"] = svg.append("g");
}

var fitnessData = {};

function fitness(id, value){
  if (!fitnessData[id]){
    fitnessData[id] = [];
    for(i = 0; i < 100; i++){
      fitnessData[id].push(0);
    }
  }
 
  fitnessData[id].shift();
  fitnessData[id].push(value);

  var data = fitnessData[id];

  var svg = d3.select('#' + id).select('svg');
  var path = svg.select('path');

  var width = $('#' + id).children().width(),
      height = $('#' + id).children().height();
  
  var y = d3.scaleLinear()
    .range([height, 0])
    .domain([-1, maxFitness]);

  var x = d3.scaleLinear()
    .range([0, width])
    .domain([data.length, 0]);


  var lineChart = d3.line()
                    .x(function(d, i){ return x(i) })
                    .y(function(d){ return y(d) });

  var yAxis = d3.axisLeft(y);
  var xAxis = d3.axisBottom(x);

  fitnessAxis[id+'y'].attr("transform", "translate(30,0)").call(yAxis);
  fitnessAxis[id+'x'].attr("transform", "translate(30,"+ (height - 19) + ")").call(xAxis);

  path.attr("transform", "translate(30,0)").attr("d", lineChart(data));
}

function getParameterByName(name) {
    var url = window.location.href;
    name = name.replace(/[\[\]]/g, "\\$&");
    var regex = new RegExp("[?&]" + name + "(=([^&#]*)|&|#|$)"),
        results = regex.exec(url);
    if (!results) return null;
    if (!results[2]) return '';
    return decodeURIComponent(results[2].replace(/\+/g, " "));
}
