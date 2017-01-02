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

}

var allNodes = {};
var allLinks = {};

function addIfNewNode(nodes, newNodes, node_id){
  if (nodes.indexOf(node_id) < 0) {
    newNodes.push({id: "node" + node_id, group: 0});
    nodes.push(node_id);
  }
}

function createLink(node_in, node_out, weight){
    return {source: "node" + node_in, target: "node" + node_out, value: weight };
}

function addIfNewLink(links, newLinks, updateLinks, node_in, node_out, weight){
  var link_id = node_in + "_" + node_out;
  if (links.indexOf(link_id) < 0) {
    newLinks.push(createLink(node_in, node_out, weight));
    links.push(link_id);
  } else {
    updateLinks.push(createLink(node_in, node_out, weight));
  }
}

function getNodesLinks(id, genes){
  var newNodes = [];
  var newLinks = [];
  var updateLinks = [];
  var removeLinks = [];
  var removeNodes = [];

  var nodes = allNodes[id];
  var links = allLinks[id];

  if (!nodes) { allNodes[id] = []; nodes = allNodes[id]; };
  if (!links) { allLinks[id] = []; links = allLinks[id]; };
 
  genes.forEach(function(gene){
    if (gene.enabled) {
      addIfNewNode(nodes, newNodes, gene.in_neuron_id);
      addIfNewNode(nodes, newNodes, gene.out_neuron_id);
      addIfNewLink(links, newLinks, updateLinks, gene.in_neuron_id, gene.out_neuron_id, gene.weight);
    }
  });

  links.forEach(function(link){
    var found = false;
    genes.forEach(function(gene){
      if (gene.in_neuron_id + "_" + gene.out_neuron_id == link) found = true;
    });
    var link_ids = link.split("_");
    var linkObj = createLink(link_ids[0], link_ids[1], 0);
    if (!found) removeLinks.push(linkObj);
  });

  nodes.forEach(function(node_id){
    var found = false;
    genes.forEach(function(gene){
      if (gene.in_neuron_id == node_id || gene.out_neuron_id == node_id) found = true;
    });
    if (!found) removeNodes.push("node" + node_id);
  });

  removeLinks.forEach(function(link){
    var linkId = link.source.split("node")[1] + "_" + link.target.split("node")[1];
    var linkPos = links.indexOf(linkId);
    links.splice(linkPos, 1);
  });

  removeNodes.forEach(function(node_id){
    nodes.splice(nodes.indexOf(node_id), 1);
  });

  return {delete:{links: removeLinks, nodes: removeNodes} , update:{links: updateLinks}, add:{nodes: newNodes, links: newLinks}};
}

function network(id, genes){
  var nodesLinksUpdate = getNodesLinks(id, genes); 
  var newNodesLinks = nodesLinksUpdate.add;
  var updateLinks = nodesLinksUpdate.update.links;
  var deleteLinks = nodesLinksUpdate.delete.links;
  var deleteNodes = nodesLinksUpdate.delete.nodes;

  var svg = d3.select('#' + id).select('svg');

  deleteLinks.forEach(function(linkDelete){
    var index = 0;
    var deleteIndex = -1;
    graph.links.forEach(function(link){
      if (linkDelete.source == link.source.id && linkDelete.target == link.target.id) {
        deleteIndex = index;
      }
      index++;
    });
    if (deleteIndex >= 0) {
      graph.links.splice(deleteIndex, 1);
    }
  });

  deleteNodes.forEach(function(nodeDelete){
    var index = 0;
    var deleteIndex = -1;
    graph.nodes.forEach(function(node){
      if (nodeDelete == node.id) {
        deleteIndex = index;
      }
      index++;
    });
    if (deleteIndex >= 0) {
      graph.nodes.splice(deleteIndex, 1);
    }
  });

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


  node = node.data(graph.nodes);
  node.exit().remove();

  node = node.enter().append("circle")
      .attr("r", 5)
      .attr("fill", function(d) { return color(d.group); })
      .merge(node)
      .call(d3.drag()
          .on("start", dragstarted)
          .on("drag", dragged)
          .on("end", dragended));

  link = link.data(graph.links)
  link.exit().remove();
  link = link.enter().append("line")
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

