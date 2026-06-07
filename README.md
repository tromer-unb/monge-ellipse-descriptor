# Monge–Ellipse Descriptor for Pore Architecture

This repository provides the Python implementation of the Monge–Ellipse Descriptor, an interpretable geometrical descriptor for segmented two-dimensional rock pore images.

The method represents each pore by classical morphometric quantities and by a second-moment ellipse. Local pore neighborhoods are defined using Delaunay triangulation of pore centroids. For each local pore triplet, direction-dependent elliptical radii are used to compute external homothetic centers, and a Monge–Ellipse score quantifies departures from ideal homothetic collinearity.

The descriptor combines pore fraction, pore abundance, pore-size statistics, elliptical anisotropy, boundary roughness, dentation, convexity loss, and local inter-pore organization into a compact image-level morphological signature.
