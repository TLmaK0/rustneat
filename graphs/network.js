var simulation;
var color;
var link;
var node;
var graph={nodes:[], links:[]};

function ticked() {
  link
      .attr("x1", function(d) { return d.source.x; })
      .attr("y1", function(d) { return d.source.y; })
      .attr("x2", function(d) { return d.target.x; })
      .attr("y2", function(d) { return d.target.y; });

  node
      .attr("cx", function(d) { return d.x; })
      .attr("cy", function(d) { return d.y; });
}

function network_init(id){
  var svg = d3.select('#' + id).append('svg');
  svg.append("g").attr("class", "nodes");
  svg.append("g").attr("class", "links");

  var width = $('#' + id).children().width(),
      height = $('#' + id).children().height();

  color = d3.scaleOrdinal(d3.schemeCategory20);

  simulation = d3.forceSimulation()
      .force("link", d3.forceLink().id(function(d) { return d.id; }))
      .force("charge", d3.forceManyBody())
      .force("center", d3.forceCenter(width / 2, height / 2))
      .alphaTarget(1).on("tick", ticked);

  link = svg.select(".links")
    .selectAll("line");

  node = svg.select(".nodes")
    .selectAll("circle");

  node.append("title")
      .text(function(d) { return d.id; });

}

var allNodes = {};
var allLinks = {};

function addIfNewNode(nodes, newNodes, node_id){
  if (!nodes[node_id]) {
    newNodes.push({id: "node" + node_id, group: 0});
    nodes[node_id] = true;
  }
}

function addIfNewLink(links, newLinks, updateLinks, node_in, node_out, weight){
  var link_id = node_in + "_" + node_out;
  if (!links[link_id]) {
    newLinks.push({source: "node" + node_in, target: "node" + node_out, value: weight });
    links[link_id] = true;
  } else {
    updateLinks.push({source: "node" + node_in, target: "node" + node_out, value: weight });
  }
}

function getNodesLinks(id, genes){
  //TODO: remove old nodes and links

  var newNodes = [];
  var newLinks = [];
  var updateLinks = [];

  var nodes = allNodes[id];
  var links = allLinks[id];

  if (!nodes) { allNodes[id] = {}; nodes = allNodes[id]; };
  if (!links) { allLinks[id] = {}; links = allLinks[id]; };
 
  genes.forEach(function(gene){
    if (gene.enabled) {
      addIfNewNode(nodes, newNodes, gene.in_neuron_id);
      addIfNewNode(nodes, newNodes, gene.out_neuron_id);
      addIfNewLink(links, newLinks, updateLinks, gene.in_neuron_id, gene.out_neuron_id, gene.weight);
    }
  });
  return {delete:{} , update:{links: updateLinks}, add:{nodes: newNodes, links: newLinks}};
}

function network(id, genes){
  var nodesLinksUpdate = getNodesLinks(id, genes); 
  var newNodesLinks = nodesLinksUpdate.add;
  var updateLinks = nodesLinksUpdate.update.links;

  var svg = d3.select('#' + id).select('svg');

  updateLinks.forEach(function(linkUpdate){
    graph.links.forEach(function(link){
      if (link.source.id == linkUpdate.source && link.target.id == linkUpdate.target){
        link.value = linkUpdate.value;
      }
    });
  });

  newNodesLinks.nodes.forEach(function(node){
    graph.nodes.push(node);
  });

  newNodesLinks.links.forEach(function(link){
    graph.links.push(link);
  });

  node = node.data(graph.nodes)
    .enter().append("circle")
      .attr("r", 5)
      .attr("fill", function(d) { return color(d.group); })
      .merge(node)
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended));

  link = link.data(graph.links)
    .enter().append("line")
      .attr("stroke-width", function(d) { return Math.sqrt(d.value * 10); })
      .merge(link);

  simulation.nodes(graph.nodes);
  simulation.force("link").links(graph.links);
  simulation.alpha(1).restart();

  function dragstarted(d) {
    if (!d3.event.active) simulation.alphaTarget(0.3).restart();
    d.fx = d.x;
    d.fy = d.y;
  }

  function dragged(d) {
    d.fx = d3.event.x;
    d.fy = d3.event.y;
  }

  function dragended(d) {
    if (!d3.event.active) simulation.alphaTarget(0);
    d.fx = null;
    d.fy = null;
  }
}

