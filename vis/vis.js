// Enable strict mode
"use strict";

// NOTE: This library uses d3.v3.js which is included in the main HTML file

/* Generic helper methods and classes here */
function Exception(string) {
    // Exception for general use
    this.string = string;
}


/* Tree specific classes and methods */

function Node(data) {
    // Node Object
    // data.labels should never be null...
    this.labels = data.labels;

    // The data we really want
    this.l_labels = data.lpart || null;
    this.l_pure = data.lpure || null;

    this.r_labels = data.rpart || null;
    this.r_pure = data.rpure || null;

    this.overlap = data.overlap || null;
    this.accuracy = data.accuracy || null;

    // Actual object based children
    this.left = null;
    this.right = null;
};

function ConstructTree(dataList) {
    // Method to create the tree. Recursive.
    var nodeData = null;
    if(dataList.length === 0) {
        // Nothing left in the list
        console.log('We\'re Done');
        return null;
    } else {
        nodeData = dataList.shift();
    }

    var root = new Node(nodeData);

    if(root.labels.length === 1) {
        // Then we are a leaf node;
        //console.log('Leaf');
        return root;
    } else {
        //console.log('Intermediate');
        // We just create and push to the left node
        root.left = ConstructTree(dataList);
        // Then do the same for the right
        root.right = ConstructTree(dataList);

        return root;
    }
};

/* D3 visualsation bits go here */

var size = {
    height: 600,
    width: 1000,
    padding: 20,
    radius: 10,
    transition: 500,
}

var d3tree = d3.layout.tree()
    .sort(null)
    .size([size.width - 2*size.padding, size.height - 2*size.padding])
    .children(function(d) {
        if(d.left === null && d.right === null) {
            return null;
        } else {
            var kids = [];
            if(d.left) {
               kids.push(d.left);
            }
            if(d.right) {
                kids.push(d.right);
            }
            return kids;
        }
    });

var svg = d3.select('#tree-container')
    .append('svg:svg')
    .attr('width', size.width)
    .attr('height', size.height);

var link = d3.svg.diagonal()
    .projection(function(d) {
        return [d.x + size.padding, d.y + size.padding]; // Is this necessary?
    });

function update(treeData) {
    var nodes = d3tree.nodes(treeData);
    var links = d3tree.links(nodes);
    //console.log(nodes);
    //console.log(links);

    var linkGroup = svg.selectAll('path.link')
        .data(links);


    // Move old links to new positions
    linkGroup.transition()
        .duration(size.transition)
        .attr('d', link)
        .call(function() {
            // Add the new links
            var newLinksGroup = linkGroup.enter()
                .insert('svg:path', '.node')
                .attr('class', 'link')
                .attr('d', function(d) {
                    var o = {x: d.source.x, y: d.source.y};
                    return link({source: o, target: o});
                })
                .transition()
                .delay(size.transition)
                .duration(size.transition)
                .attr('d', link);
        });

    var nodeGroup = svg.selectAll('g.node')
        .data(nodes);


    // Move the old nodes to their new positions
    nodeGroup.transition()
        .duration(size.transition)
        .attr('transform', function(d) {
            return 'translate(' + (d.x + size.padding) + ',' + (d.y + size.padding) + ')';
        })
        .call(function() {
            // Add nodes to the vis
            var newNodesGroup = nodeGroup.enter()
                .append('svg:g')
                .attr('class', 'node')
                .attr('transform', function(d) {
                    var initx = null;
                    var inity = null;

                    if(!d.parent) {
                        initx = d.x + size.padding;
                        inity = d.y + size.padding;
                    } else {
                        initx = d.parent.x + size.padding;
                        inity = d.parent.y + size.padding;
                    }
                    return 'translate(' + initx + ',' + inity + ')';
                })
                .on('mouseover', function(d) {
                    d3.select('#labels').html(d.labels);
                    d3.select('#lpart-text').html(d.l_labels);
                    d3.select('#rpart-text').html(d.r_labels);
                    d3.select('#overlap').html(d.overlap);
                    d3.select('#accuracy').html(d.accuracy);
                })
                .on('mouseout', function(d) {
                    d3.select('#labels').html('');
                    d3.select('#lpart-text').html('');
                    d3.select('#rpart-text').html('');
                    d3.select('#overlap').html('');
                    d3.select('#accuracy').html('');
                });


            newNodesGroup.transition()
                .delay(size.transition)
                .duration(size.transition)
                .attr('transform', function(d) {
                    return 'translate(' + (d.x + size.padding) + ',' + (d.y + size.padding) + ')';
                });


            newNodesGroup.append('svg:circle')
                .attr('class', 'node-dot')
                .attr('r', 0)
                .transition()
                .delay(size.transition)
                .duration(size.transition)
                .attr('r', size.radius);
        });
}


/* Polling and related bits */

function pollHandler(error, contents) {
    if (error) {
        console.log('Could not fetch data, retrying in 10 seconds...');
    } else {
        data = contents.toString().split('\n')
            .filter(function(v) {
                // filter out empty strings/blank lines
                return v.length > 0
            })
            .map(JSON.parse);
        var tree = ConstructTree(data);
        update(tree);
    }
};

// Should this be global/ near global?
var data = null;

function poller() {
    // Poll handling function
    d3.text('./live.json', pollHandler);
    setTimeout(poller, 5000);
};

// Kick everything off
poller();


/* Node specific stuff */
    //var util = require('util');
    //var fs = require('fs');
    //console.log(util.inspect(tree, {showHidden: false, depth: null}));
    //fs.readFile('myfile', callbackFunc);
