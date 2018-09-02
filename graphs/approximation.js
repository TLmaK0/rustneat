var svg, paths = []; 

var lineChart;

function approximation_init(id){
  svg = d3.select('#' + id).append('svg');
  var width = $('#' + id).children().width() / 2;
  var rangeX = [-width, width];
  var height = $('#' + id).children().height();
  var rangeY = [0, height];

  for(var n=0; n<10; n++ ) {
    paths[n] = svg.append('path');
    paths[n].attr("transform", "translate(" + width + ")");
    paths[n].attr("stroke", "rgb(" + (50 + n * 20) + ",0,0)");
  }
  paths[0].attr("stroke", "rgb(200,200,200)");
  scaleX = d3.scaleLinear().domain([-10, 10]).range(rangeX);
  scaleY = d3.scaleLinear().domain([0, 100]).range(rangeY);

  lineChart = d3.line()
                    .x(function(d){ return scaleX(d[0]); })
                    .y(function(d){ return height - scaleY(d[1]); });
}

var functionToApproximate = [];
var functionApproximations = [];

function approximation(id, value){
  if (functionToApproximate.length == 0) functionToApproximate = value;
  if (functionApproximations.length == 9) functionApproximations.shift();

  functionApproximations.push(value);

  paths[0].attr("d", lineChart(functionToApproximate));

  for(var n=0; n < functionApproximations.length; n++ ){
    paths[n + 1].attr("d", lineChart(functionApproximations[n]));
  }
}
