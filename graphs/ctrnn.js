var svg, path, values = [];

function ctrnn_init(id){
  svg = d3.select('#' + id).append('svg');
  var width = $('#' + id).children().width() / 2;
  var rangeX = [-width, width];
  var height = $('#' + id).children().height();
  var rangeY = [0, height];

  path = svg.append('path');
  path.attr("transform", "translate(" + width + ")");
  path.attr("stroke", "rgb(255,0,0)");

  scaleX = d3.scaleLinear().domain([0, 100]).range(rangeX);
  scaleY = d3.scaleLinear().domain([10, -10]).range(rangeY);

  lineChart = d3.line()
                    .x(function(d){ return scaleX(d[0]); })
                    .y(function(d){ return height - scaleY(d[1]); });

  var yAxis = d3.axisLeft(scaleY);
  svg.append("g").attr("transform", "translate(30,0)").call(yAxis);
}

function ctrnn(id, value){
  var neuron1_value = value[1];
  values.push([values.length, neuron1_value]);
  path.attr("d", lineChart(values));
}
