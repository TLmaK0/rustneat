function distance(id, data){
  var svg = d3.select(id),
      width = $(id).width(),
      height = $(id).height();

  var x = d3.scaleLinear()
    .range([0, width])
    .domain([0, 1]);
  var y = d3.scaleBand().domain(data.map(function(d){return d.id;})).range([0,height]);

  var bars = svg.selectAll("rect")
    .data(data);

  bars.enter().append("rect");

  bars.attr("x", x(0))
    .attr('y', function(d){ return y(d.id); })
    .attr('height', function(d){ return y.bandwidth(); })
    .attr('width', function(d){ return x(d.v);}); 
}
