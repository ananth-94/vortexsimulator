#ifndef NODE_H
#define NODE_H

#include <iostream>
#include <vector>
#include <cmath>
#include <utility> // For pair function
#include <bits/stdc++.h>
#include "parameters.h"

using namespace std;

// A class defining a Barnes-Hut node
class node
{
	private:
		// Data associated with every node object
		double size_bh; // Size of the root node
		int level; // Level of the node. Root would be 0. First subdivision would be 1. And so on
		vector<double> anchor; // Anchor point of the node // Centre of the box // Used as origin for calculations within this node

		double nodeSize; // Size of this node
		double node_xmin, node_xmax, node_ymin, node_ymax; // Limits of this node
		
		vector<double> body_within; // Condition to track if this node contains a body
		bool body_within_state = false;
		
		unique_ptr<node> children[4] = {nullptr}; // A node can have four children nodes

		double total_mass = 0; // Total mass of bodies within the node // Mass is set to 1 // Thus total number of bodies within the node
		vector<double> centre_of_mass = {0,0}; // Centre of mass of the bodies in this node

	public:
		// Constructor
		node(double size_bh_box, int value_level = 0, vector<double> value_anchor = {0,0})
		{
			// Initialize node
			size_bh = size_bh_box;
			level = value_level;
			anchor = value_anchor;

			// Define the boundaries of the node
			nodeSize = (1/pow(2,level)) * size_bh;
			node_xmin = anchor[0] - nodeSize/2;
			node_xmax = anchor[0] + nodeSize/2;
			node_ymin = anchor[1] - nodeSize/2;
			node_ymax = anchor[1] + nodeSize/2;
			
			// Initialize centre of mass to the anchor position
			centre_of_mass = anchor;
		};

		// Destructor
		~node()
		{
			// Destruction of dynamically allocated memory handled by unique_ptr 
		};

		// Inserting a body into the node
		void insert(vector<double> body)
		{	
			// If the body is within the node
			if (body[0] > node_xmin && body[0] < node_xmax && body[1] > node_ymin && body[1] < node_ymax)
			{
				// If no other body exists there				
				if (body_within_state == false)
				{		
					body_within_state = true;
					body_within = body;
				}
				
				// If another body exists there					
				else if (body_within_state == true)
				{
					// And no division has been done
					if (children[0] == nullptr)
					{		
						// Subdivide the node into four subnodes
						for (int index = 0; index < 4 ; index++)
						{
							children[index] = make_unique<node>(size_bh, level+1, calculate_anchor(anchor, index));	
						}
						
						// Insert the old body in the appropriate child and then break out of the loop
						for (int index = 0; index < 4 ; index++)
						{
							if (body_within[0] > (*children[index]).node_xmin && body_within[0] < (*children[index]).node_xmax && body_within[1] > (*children[index]).node_ymin && body_within[1] < (*children[index]).node_ymax)
							{
								(*children[index]).insert(body_within);
								break;
							}
						}

						// Insert new body in the appropriate child and then break out of the loop
						for (int index = 0; index < 4 ; index++)
						{
							if (body[0] > (*children[index]).node_xmin && body[0] < (*children[index]).node_xmax && body[1] > (*children[index]).node_ymin && body[1] < (*children[index]).node_ymax)
							{
								(*children[index]).insert(body);
								break;
							}
						}
					}

					// And division has been done
					else if (children[0] != nullptr)
					{
						// Simply try inserting into the subnodes
						for (int index = 0; index < 4 ; index ++)
						{
							if (body[0] > (*children[index]).node_xmin && body[0] < (*children[index]).node_xmax && body[1] > (*children[index]).node_ymin && body[1] < (*children[index]).node_ymax)
							{
								(*children[index]).insert(body);
								break;
							}
						}
					}
				}
				
				// Update center_of_mass and total_mass of this node
				// Mass of every vortex is taken to be 1				
				++total_mass; // Mass increment is by 1
				centre_of_mass[0] = (((total_mass-1) * centre_of_mass[0]) + (1 * body[0])) / (total_mass);
				centre_of_mass[1] = (((total_mass-1) * centre_of_mass[1]) + (1 * body[1])) / (total_mass);
			}
		}

		// Calculate anchor of children from that of parent
		vector<double> calculate_anchor(vector<double> anchor_old, int index)
		{
			vector<double> anchor_new(2, 0.0);
			
			if (index == 0)
			{
				anchor_new[0] = anchor_old[0] + (nodeSize/4);
				anchor_new[1] = anchor_old[1] + (nodeSize/4);
			}
			else if (index == 1)
			{
				anchor_new[0] = anchor_old[0] - (nodeSize/4);
				anchor_new[1] = anchor_old[1] + (nodeSize/4);
			}
			else if (index == 2)
			{
				anchor_new[0] = anchor_old[0] - (nodeSize/4);
				anchor_new[1] = anchor_old[1] - (nodeSize/4);
			}
			else if (index == 3)
			{
				anchor_new[0] = anchor_old[0] + (nodeSize/4);
				anchor_new[1] = anchor_old[1] - (nodeSize/4);
			}

			return anchor_new;
		}
		
		// Compute force on a vortex placed into the node // Usually invoked by the root node
		vector<double> compute_force(vector<double> body, double theta)
		{	
			// Compute only if there exists a body within the node, of course
			if (body_within_state == false)
			{
				return{0,0};
			}
			
			double dx = body[0] - centre_of_mass[0];
			double dy =  body[1] - centre_of_mass[1];
			double dist = sqrt(dx * dx + dy * dy);
			
			// Force on body due to itself should not be included
			if (dist == 0)
			{
				return{0,0};
			}
			
			// If the body and the node/subnode satisfy the theta condition, calculate force and break from loop
			else if  ((nodeSize / dist) < theta || children[0] == nullptr)
			{
				double forcex = -(total_mass * dy) / (dist * dist);
				double forcey =  (total_mass * dx) / (dist * dist);
				return {forcex, forcey};
			}
			
			// If they don't, then subdivide and recheck
			else 
			{
				vector<double> force(2,0.0);
				for (int i = 0; i < 4; i++)
				{
					vector<double> temp_force = (*children[i]).compute_force(body, theta);
					force[0] += temp_force[0];
					force[1] += temp_force[1];
				}
				return force;
			}
		}
		
		// Print the centre of masses of the leaves 
		// Useful for debugging
		void print_cm_leaves()
		{
			if (children[0] == nullptr)
			{
				if (body_within_state == true)
				{
					cout << centre_of_mass[0] << ", " << centre_of_mass[1] << endl;
					cout << "total_mass = " << total_mass << endl;
				}
			}
			else
			{
				for (int i = 0; i <4; i++)
				{
					(*children[i]).print_cm_leaves();
				}
			}
		}
		
	};

#endif
