// Enable strict mode
//"use strict";

var raw_data = [
['1','0','3','2','5','4','7','6','9','8'], 
['1','3','2','4','7','6','9','8'], 
['1','3','2','7','6','9','8'], 
['1','3','2','7','9','8'], 
['9','3','7'], 
['9','7'], 
['9'], 
['7'], 
['3','7'], 
['3'], 
['7'], 
['1','2','7','9','8'], 
['9','2','7'], 
['9','7'], 
['9'], 
['7'], 
['2'], 
['1','7','9','8'], 
['1','8','7'], 
['1'], 
['8','7'], 
['8'], 
['7'], 
['9','8','7'], 
['9','7'], 
['9'], 
['7'], 
['8'], 
['1','2','6','8'], 
['1','8','2'], 
['1','8'], 
['1'], 
['8'], 
['2'], 
['6'], 
['1','2','4','7','6'], 
['1','2','4','7'], 
['1','4','7'], 
['1','7'], 
['1'], 
['7'], 
['4','7'], 
['4'], 
['7'], 
['2','7'], 
['2'], 
['7'], 
['6'], 
['1','0','3','2','5','7','9','8'], 
['1','3','2','5','7','9','8'], 
['1','3','2','5','7','9'], 
['1','9','3','2','7'], 
['1'], 
['9','3','2','7'], 
['9','3','7'], 
['9','7'], 
['9'], 
['7'], 
['3','7'], 
['3'], 
['7'], 
['2','7'], 
['2'], 
['7'], 
['9','5','7'], 
['9','7'], 
['9'], 
['7'], 
['5'], 
['1','2','7','9','8'], 
['9','2','7'], 
['9','7'], 
['9'], 
['7'], 
['2'], 
['1','7','9','8'], 
['1','8','7'], 
['1'], 
['8','7'], 
['8'], 
['7'], 
['9','8','7'], 
['9','7'], 
['9'], 
['7'], 
['8'], 
['1','0','2','7','9','8'], 
['1','9','2','7','8'], 
['9','2','7'], 
['9','7'], 
['9'], 
['7'], 
['2'], 
['1','7','9','8'], 
['1','8','7'], 
['1'], 
['8','7'], 
['8'], 
['7'], 
['9','8','7'], 
['9','7'], 
['9'], 
['7'], 
['8'], 
['0'], 
];

function Exception(string) {
    this.string = string;
}

function Node(labels) {
    this.labels = null;
    this.left = null;
    this.right = null;
    this.l_labels = null;
    this.r_labels = null;
    this.overlap = null;

    if (typeof labels !== undefined) {
        this.labels = labels
    }
};

function ConstructTree(raw_data) {
    var label_list = null;
    if(raw_data.length === 0) {
        console.log('We\'re Done');
    } else {
        label_list = raw_data.shift();
    }

    //console.log(label_list);

    var root = new Node(label_list);

    //console.log(root);

    
    if(label_list.length === 1) {
        // Then we are a leaf node;
        //console.log('Leaf');
        return root;
    } else {
        //console.log('Intermediate');
        // We just create and push to the left node
        root.left = ConstructTree(raw_data);
        // Then do the same for the right
        root.right = ConstructTree(raw_data);

        return root;
    }
};

var tree = ConstructTree(raw_data);
console.log(tree);

// Node specific stuff
//var util = require('util');
//console.log(util.inspect(tree, {showHidden: false, depth: null}));

var size = {
    height: '600',
    width: '1000',
    circle_radius: 10,
    padding: 20,
};

var d3tree = d3.layout.tree()
    .sort(null)
    .size([size.width - 2*size.padding, size.height - 2*size.padding])
    .children(function(d) {
        if(d.left === null && d.right === null) {
            return null;
        } else {
            return [d.left, d.right];
        }
    });

var nodes = d3tree.nodes(tree);
var links = d3tree.links(nodes);

// var layout = d3.select('body')
//      .append("svg:svg").attr("width", size.width).attr("height", size.height)
//      .append("svg:g")
//      .attr("class", "container");
var layout = d3.select('#tree-container')
     .append("svg:svg")
     .attr("width", size.width)
     .attr("height", size.height);

var link = d3.svg.diagonal()
    .projection(function(d) {
        return [d.x, d.y];
    });

var linkgroup = layout.selectAll('path.link')
    .data(links)
    .enter()
    .append('svg:path')
    .attr('class', 'link')
    .attr('d', link)
    .attr('transform', function(d) {
        return 'translate(' + size.padding + ',' + size.padding + ')';
    });

var nodeGroup = layout.selectAll('g.node')
    .data(nodes)
    .enter()
    .append('avg:g')
    .attr('class', 'node')
    .attr('transform', function(d) {
        return 'translate(' + (d.x + size.padding) + ',' + (d.y + size.padding) + ')';
    })
    .on('mouseover', function(d) {
        d3.select('#labels')
            .html(d.labels);
        d3.select('#lchild')
            .html(d.l_labels);
        d3.select('#rchild')
            .html(d.r_labels);
        d3.select('#overlap')
            .html(d.overlap);
    })
    .on('mouseout', function(d) {
        d3.select('#labels')
            .html('');
        d3.select('#lchild')
            .html('');
        d3.select('#rchild')
            .html('');
        d3.select('#overlap')
            .html('');
    });

nodeGroup.append('svg:circle')
    .attr('class', 'node-dot')
    .attr('r', size.circle_radius);
