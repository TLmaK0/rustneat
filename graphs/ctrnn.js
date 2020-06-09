var svg, paths = [], values = [];
var posx = 0;
var currentGraph = 0;
var colors = ['black', 'blue', 'red', 'green', 'grey', 'magenta']

function ctrnn_init(id){
  svg = d3.select('#' + id).append('svg');
  var width = $('#' + id).children().width();
  var rangeX = [0, width];
  var height = $('#' + id).children().height();
  var rangeY = [0, height];

  for(var n=0; n<6; n++ ) {
    paths[n] = svg.append('path');
    paths[n].attr("stroke", colors[n]);
  }

  scaleX = d3.scaleLinear().domain([0, 16]).range(rangeX);
  scaleY = d3.scaleLinear().domain([2.5, -1.5]).range(rangeY);

  lineChart = d3.line()
                    .x(function(d){ return scaleX(d[0]); })
                    .y(function(d){ return scaleY(d[1]); });

  var yAxis = d3.axisLeft(scaleY);
  var xAxis = d3.axisBottom(scaleX);
  svg.append("g").attr("transform", "translate(30,0)").call(yAxis);
  svg.append("g").attr("transform", "translate(30," + (height - 20) + ")").call(xAxis);

  values[0] = [];
}

function ctrnn(id, value){
  if (value == 'reset') return;
  for(var n = 0; n < value.length; n++){
    values[currentGraph + n].push([posx, value[n]]);
    paths[currentGraph + n].attr("transform", "translate(30,0)").attr("d", lineChart(values[currentGraph + n]));
  }
  posx++;
}

function reset(){
  posx = 0;
  currentGraph++;
  values[currentGraph] = [];
  return 'reset';
}
