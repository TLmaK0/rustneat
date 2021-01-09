var maxFitness = getParameterByName('max_fitness');

function getParameterByName(name) {
    var url = window.location.href;
    name = name.replace(/[\[\]]/g, "\\$&");
    var regex = new RegExp("[?&]" + name + "(=([^&#]*)|&|#|$)"),
        results = regex.exec(url);
    if (!results) return null;
    if (!results[2]) return '';
    return decodeURIComponent(results[2].replace(/\+/g, " "));
}

var data = [];

var tmpData = [];

var speciesKeys = [];

var speciesColor = {};

var color = d3.scaleSequential().domain([0,maxFitness]).interpolator(d3.interpolateInferno);

var maxData  = 100;

var graphs = {};

var update = 0;

var updateEvery = 20;

function species_init(id){
  var svg = d3.select('#' + id).append('svg');
  graphs['graph'+id] = svg.append('g');
  graphs['leftAxis'+id] = svg.append('g');
  graphs['bottomAxis'+id] = svg.append('g');
}

function species(id, specieData){
  insertNewData(specieData);
  update++;
  if (update < updateEvery) return;

  update = 0;
  var graph = graphs['graph'+id];
  var leftAxis = graphs['leftAxis'+id]; 
  var bottomAxis = graphs['bottomAxis'+id]; 

  var stackChart = d3.stack()
    .keys(speciesKeys).value(function(d, key){ 
      if (d[key]) return d[key].organisms; 
      return 0;
    })(data);

  x = d3.scaleLinear()
        .domain([0, data.length]).
        range([0, 600])
  y = d3.scaleLinear()
        .domain([0, 200])
        .range([0, 300]);

  var yAxis = d3.axisLeft(y);

  leftAxis.call(yAxis);

  graph.selectAll('path')
    .remove()
    .exit()
    .data(stackChart).enter().append('path')
    .attr('fill', function(d) { return speciesColor[d.key]; })
    .attr('stroke', '#ffffff')
    .attr('d', d3.area()
                    .x((d, i) => x(i))
                    .y0((d) => y(d[0]))
                    .y1((d) => y(d[1])));
}

function insertNewData(specieData){
  tmpData.push(specieData);

  speciesColor[specieData.id] = color(specieData.fitness);

  if (tmpData.length > maxData * speciesKeys.length) tmpData = tmpData.slice(maxData);

  if (!speciesKeys.includes(specieData.id)) speciesKeys.push(specieData.id);

  data = Array.from(d3.group(tmpData, d => d.timestamp), ([key, value]) => value);
}

