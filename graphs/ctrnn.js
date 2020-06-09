var svg, paths = [], values = [];
var colors = ['black', 'blue', 'red', 'green', 'grey', 'magenta']
var xAxisBar, yAxisBar;

function ctrnn_init(id){
}

function clear(id, domainX, domainY){
  posx = 0;
  currentGraph = 0;
  values = [];
  for(var n = 0; n < colors.length; n++) values[n] = [];

  if (svg) svg.remove();
  svg = d3.select('#' + id).append('svg');
  var height = $('#' + id).children().height();

  for(var n=0; n < colors.length; n++ ) {
    paths[n] = svg.append('path');
    paths[n].attr("stroke", colors[n]);
  }

  lineChart = d3.line()
  xAxisBar = svg.append("g").attr("transform", "translate(30," + (height - 20) + ")");
  yAxisBar = svg.append("g").attr("transform", "translate(30,0)");


  scaleX(id, domainX);
  scaleY(id, domainY);
}

function scaleY(id, domainY){
  var height = $('#' + id).children().height();
  var rangeY = [0, height];
  scaleY = d3.scaleLinear().domain([domainY, -domainY]).range(rangeY);
  lineChart.y(function(d){ return scaleY(d[1]); });
  var yAxis = d3.axisLeft(scaleY);
  yAxisBar.call(yAxis);
}

function scaleX(id, domainX){
  var width = $('#' + id).children().width();
  var rangeX = [0, width];
  scaleX = d3.scaleLinear().domain([0, domainX]).range(rangeX);
  var xAxis = d3.axisBottom(scaleX);
  lineChart.x(function(d){ return scaleX(d[0]); })
  xAxisBar.call(xAxis);
}

function ctrnn(id, value){
  if (!Array.isArray(value)) return; //method calls
  for(var n = 0; n < value.length; n++){
    values[currentGraph + n].push([posx, value[n]]);
    paths[currentGraph + n].attr("transform", "translate(30,0)").attr("d", lineChart(values[currentGraph + n]));
  }
  posx++;
}

function next_graph(){
  posx = 0;
  currentGraph++;
  values[currentGraph] = [];
}
