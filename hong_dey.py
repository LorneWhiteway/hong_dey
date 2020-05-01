#!/usr/bin/env python




# See my notes p. HD 50
def scale_filter_condition(filter_condition, scale):

	return (0.5 * (filter_condition[0] * (1.0 + scale) + filter_condition[1] *(1.0 - scale)), 0.5 * (filter_condition[0] * (1.0 - scale) + filter_condition[1] *(1.0 + scale)),
		0.5 * (filter_condition[2] * (1.0 + scale) + filter_condition[3] *(1.0 - scale)), 0.5 * (filter_condition[2] * (1.0 - scale) + filter_condition[3] *(1.0 + scale)),
		0.5 * (filter_condition[4] * (1.0 + scale) + filter_condition[5] *(1.0 - scale)), 0.5 * (filter_condition[4] * (1.0 - scale) + filter_condition[5] *(1.0 + scale)))


# z_mean, use_z_spec and run_num used only for certain configurations. Precisely one of target_average_edges_per_vertex and linking_length should be specified.
def hong_dey_run_configuration(configuration, target_average_edges_per_vertex = None, linking_length = None, z_mean = 0.0, use_z_spec = True, run_num = 0):

	data_directory = '/share/splinter/ucapwhi/cosmic_web/'
	
	filter_condition_scale = filter_condition_scale_default()

	if configuration == 0:
		# VERY FAST TEST SUITE
		catalogue_file_name_suffix = "input/buzzard_v2/buzzard-v1.1-y1a1_run_redmapper_v6.4.4_zspec_vl04_redmapper_central_and_redmagic.fits.gz"
		id_field_name = "id"
		position_specification = "RA_DEC_REDSHIFT"
		position_field_names = ("ra", "dec", "z_spec")
		filter_condition = (60.0, 62.5, -52.5, -50.0, 0.45, 0.475)
		filter_condition_scale = 1.95
		output_file_name_suffix = "hong_dey/output/fast_test_set.hdo"
	elif configuration == 1:
		# See p HD 73. Used when testing linking length sensitivity
		catalogue_file_name_suffix = "input/buzzard_v2/buzzard-v1.1-y1a1_run_redmapper_v6.4.4_zspec_vl04_redmapper_central_and_redmagic.fits.gz"
		id_field_name = "id"
		position_specification = "RA_DEC_REDSHIFT"
		position_field_names = ("ra", "dec", "z_spec")
		filter_condition = (60.0, 65.0, -55.0, -50.0, 0.45, 0.50)
		filter_condition_scale = 1.4
		output_file_name_suffix = "hong_dey/output/linking_length_sensitivity.hdo." + str(int(2.0 * linking_length)).zfill(2)
	elif configuration == 2:
		# Millennium survey data
		catalogue_file_name_suffix = "input/millennium/millennium_data_000.npy"
		id_field_name = "sequential"
		position_specification = "X_Y_Z"
		position_field_names = ("X", "Y", "Z")
		filter_condition = (120.0, 140.0, 120.0, 140.0, 120.0, 140.0)
		output_file_name_suffix = "hong_dey/output/millennium/millennium" + str(int(target_average_edges_per_vertex)).zfill(2) + ".hdo"
	elif configuration == 3:
		# From the very original investigations
		catalogue_file_name_suffix = "input/drgk_example/sva1_gold_1.0.2_run_redmapper_v6.3.3_redmagic_0.5-10.fit"
		id_field_name = "coadd_objects_id"
		position_specification = "RA_DEC_REDSHIFT"
		position_field_names = ("ra", "dec", "z")
		filter_condition = (65.0, 75.0, -60.0, -55.0, 0.50, 0.55)
		output_file_name_suffix = "hong_dey/output/output.hdo"
	elif configuration == 4:
		# Buzzard simulated data, used in preparing slides for 11 Feb 2016 telecon
		catalogue_file_name_suffix = "input/buzzard_v1/buzzard-v1.1-y1a1_run_redmapper_v6.4.4_redmagic_0.5-10.fit"
		id_field_name = "coadd_objects_id"
		position_specification = "RA_DEC_REDSHIFT"
		position_field_names = ("ra", "dec", "zspec")
		filter_condition = (45.0, 75.0, -54.0, -44.0, 0.0, 2.0)
		output_file_name_suffix = "hong_dey/output/output.hdo"
	elif configuration == 5:
		# A file containing redmapper central galaxies and redmagic galaxies, as suggested by Peter in 11 Feb 2016 telecon. Produced by LW on 10 Mar 2016.
		catalogue_file_name_suffix = "input/buzzard_v2/buzzard-v1.1-y1a1_run_redmapper_v6.4.4_zspec_vl04_redmapper_central_and_redmagic.fits.gz"
		id_field_name = "id"
		position_specification = "RA_DEC_REDSHIFT"
		position_field_names = ("ra", "dec", "zspec")
		filter_condition = (45.0, 75.0, -54.0, -44.0, 0.0, 2.0) # 'Minimal masked' large data set ...
		output_file_name_suffix = "hong_dey/output/buzzard-v1.1-y1a1_run_redmapper_v6.4.4_zspec_vl04_redmapper_central_and_redmagic.hdo"
	elif configuration == 6:
		# Preparation for telecon on 27 April 2016. Needs to know z_mean and use_z_spec.
		catalogue_file_name_suffix = "input/buzzard_v2/buzzard-v1.1-y1a1_run_redmapper_v6.4.4_zspec_vl04_redmapper_central_and_redmagic.fits.gz"
		id_field_name = "id"
		position_specification = "RA_DEC_REDSHIFT"
		position_field_names = ("ra", "dec", ("z_spec" if use_z_spec else "z"))
		filter_condition = (45.0, 75.0, -54.0, -44.0, z_mean - 0.015, z_mean + 0.015) # Slice of 'Minimal masked' large data set ...
		output_file_name_suffix = "hong_dey/output/20160427/Network_" + ("Z_SPEC" if use_z_spec else "ZDs") + "_z" + ("%.3f" % (z_mean)) + ".hdo"
	elif configuration == 7:
		# For use with specific geometry.
		catalogue_file_name_suffix = "input/specific_geometry/test10.fits"
		id_field_name = "sequential"
		position_specification = "X_Y_Z"
		position_field_names = ("X", "Y", "Z")
		filter_condition = (-100.0, 200.0, -100.0, 200.0, -100.0, 200.0,) # Everything...
		output_file_name_suffix = "hong_dey/output/specific_geometry/test_10_" + str(int(target_average_edges_per_vertex)).zfill(2) + ".hdo"
	elif configuration == 8:
		# Similar to the analysis done for the telecon on 27 April 2016, but now using Y1A1 data.
		catalogue_file_name_suffix = "input/y1a1/joint_y1_redmapper_redmagic.fits"
		id_field_name = "id"
		position_specification = "RA_DEC_REDSHIFT"
		position_field_names = ("ra", "dec", ("z_spec" if use_z_spec else "z"))
		filter_condition = (45.0, 75.0, -54.0, -44.0, z_mean - 0.015, z_mean + 0.015) # Slice of 'Minimal masked' large data set ...
		output_file_name_suffix = "hong_dey/output/20160509/Network_" + ("Z_SPEC" if use_z_spec else "Z_PHOTO") + "_z" + ("%.3f" % (z_mean)) + ".hdo"
	elif configuration == 9:
		# Sloan great wall data from cosmodb. See my notes p. HD 137-9.
		catalogue_file_name_suffix = "input/sdss/SGW.fits"
		id_field_name = "ID"
		position_specification = "X_Y_Z"
		position_field_names = ("X", "Y", "Z")
		filter_condition = (-1000.0, 1000.0, -1000.0, 1000.0, -1000.0, 1000.0) # Everything...
		output_file_name_suffix = "hong_dey/output/sdss/sdss" + str(int(target_average_edges_per_vertex)).zfill(2) + ".hdo"
	elif configuration == 10:
		# Sloan great wall data from table2i.
		catalogue_file_name_suffix = "input/sdss/J_A+A_514_A102_table2i.dat.gz.fits.gz"
		id_field_name = "Ngal"
		position_specification = "RA_DEC_DISTANCE"
		position_field_names = ("RAdeg", "DEdeg", "Dist")
		filter_condition = (150.0, 220.0, -4.0, 8.0, 197.0, 245.0)
		output_file_name_suffix = "hong_dey/output/sdss_table2i/sdss" + str(int(target_average_edges_per_vertex)).zfill(2) + ".hdo"
	elif configuration == 11:
		# sphereX data from cosmodb. See my notes p. HD 151.
		sphere_num = str(5)
		catalogue_file_name_suffix = "input/sdss/sphere" + sphere_num + ".fits"
		id_field_name = "ID"
		position_specification = "X_Y_Z"
		position_field_names = ("X", "Y", "Z")
		filter_condition = (-1000.0, 1000.0, -1000.0, 1000.0, -1000.0, 1000.0) # Everything...
		output_file_name_suffix = "hong_dey/output/sphere" + sphere_num + "/sdss" + str(int(target_average_edges_per_vertex)).zfill(2) + ".hdo"
	elif configuration == 12:
		# For the cosmic web conference - degrading the photo-z error. Needs 'run num' as an input.
		catalogue_file_name_suffix = "input/20160609/joint_y1_redmapper_redmagic_degraded_walk.fits.gz"
		id_field_name = "id"
		position_specification = "RA_DEC_REDSHIFT"
		redshift_field_names = ["zspec", "zphoto_0.1", "zphoto_0.2", "zphoto_0.3", "zphoto_0.4", "zphoto_0.5", "zphoto_0.6", "zphoto_0.7", "zphoto_0.8", "zphoto_0.9", "zphoto_1.0"]
		position_field_names = ("ra", "dec", redshift_field_names[run_num])
		filter_condition = (45.0, 75.0, -54.0, -44.0, 0.45, 0.50) # Slice of 'Minimal masked' large data set ...
		output_file_name_suffix = "hong_dey/output/20160609/test_" + str(run_num).zfill(2) + ".hdo"
	elif configuration == 13:
		# Similar to configuration 8, but for a slice 0.45 - 0.50. This is for slide 10 for the Cosmic Web conference (9 June 2016).
		catalogue_file_name_suffix = "input/y1a1/joint_y1_redmapper_redmagic.fits"
		id_field_name = "id"
		position_specification = "RA_DEC_REDSHIFT"
		position_field_names = ("ra", "dec", "z")
		filter_condition = (45.0, 75.0, -54.0, -44.0, 0.45, 0.50) # Slice of 'Minimal masked' large data set ...
		output_file_name_suffix = "hong_dey/output/20160609/slide10.hdo"
	elif configuration == 14:
		# See p. HD 190. Used to generate values for first weak lensing trials.
		catalogue_file_name_suffix = "input/WL/joint_y1_redmapper_redmagic_degraded.fits.gz"
		id_field_name = "id"
		position_specification = "RA_DEC_REDSHIFT"
		position_field_names = ("ra", "dec", "zspec")
		filter_condition = (45.0, 75.0, -54.0, -44.0, 0.45, 0.50) # Slice of 'Minimal masked' large data set ...
		output_file_name_suffix = "hong_dey/output/WL/joint_y1_redmapper_redmagic_degraded.hdo"
	elif configuration == 15:
		# From 28 Nov 2016 - run done for DK in preparation for his Belfast presentation.
		catalogue_file_name_suffix = "input/20161128/joint_bcc_mock1_redmapper_redmagic.fits"
		id_field_name = "id"
		position_specification = "RA_DEC_REDSHIFT"
		position_field_names = ("ra", "dec", "zspec")
		filter_condition = (45.0, 75.0, -54.0, -44.0, 0.45, 0.50) # Slice of 'Minimal masked' large data set ...
		output_file_name_suffix = "hong_dey/output/20161128/joint_bcc_mock1_redmapper_redmagic_" + str(int(target_average_edges_per_vertex)).zfill(2) + ".hdo"
	elif configuration == 16:
		# From 9 Feb 2017 - run done for DK for use in the filament comparison project.
		catalogue_file_name_suffix = "input/buzzard_v2/buzzard-v1.1-y1a1_run_redmapper_v6.4.4_zspec_vl04_redmapper_central_and_redmagic.fits.gz"
		id_field_name = "id"
		position_specification = "RA_DEC_REDSHIFT"
		position_field_names = "(ra,dec,z_spec)"
		filter_condition = (45.0, 75.0, -54.0, -44.0, 0.45, 0.50) # Slice of 'Minimal masked' large data set ...
		output_file_name_suffix = "hong_dey/output/20170209/buzzard-v1.1-y1a1_run_redmapper_v6.4.4_zspec_vl04_redmapper_central_and_redmagic.hdo"
	elif configuration == 17:
		# As 16, but photoz.
		catalogue_file_name_suffix = "input/buzzard_v2/buzzard-v1.1-y1a1_run_redmapper_v6.4.4_zspec_vl04_redmapper_central_and_redmagic.fits.gz"
		id_field_name = "id"
		position_specification = "RA_DEC_REDSHIFT"
		position_field_names = "(ra,dec,z)"
		filter_condition = (45.0, 75.0, -54.0, -44.0, 0.45, 0.50) # Slice of 'Minimal masked' large data set ...
		output_file_name_suffix = "hong_dey/output/20170209/buzzard-v1.1-y1a1_run_redmapper_v6.4.4_zphoto_vl04_redmapper_central_and_redmagic.hdo"
		
	if (target_average_edges_per_vertex is None) == (linking_length is None):
		raise SystemError("Exactly one of target_average_edges_per_vertex and linking_length should be specified")
		
	if target_average_edges_per_vertex is None:
		graph_size_target = linking_length
		graph_size_target_is_average_edges_per_vertex = False
	else:
		graph_size_target = target_average_edges_per_vertex
		graph_size_target_is_average_edges_per_vertex = True
		
	
	hong_dey_run(
		(data_directory + catalogue_file_name_suffix),
		structure_file_name_default(),
		id_field_name,
		position_specification,
		str(position_field_names),
		str(filter_condition),
		filter_condition_scale,
		(data_directory + output_file_name_suffix),
		graph_size_target_is_average_edges_per_vertex,
		graph_size_target,
		alpha_default(),
		filament_object_percentile_default(),
		cluster_object_percentile_default(),
		high_wcl_percentile_default()
		)
	

# Default values for interface	

def filter_condition_scale_default():
	return 1.0

def graph_size_target_is_average_edges_per_vertex_default():
	return True
	
def graph_size_target_default():
	return 6.0

def alpha_default():
	return 1.0
	
def filament_object_percentile_default():
	return 0.90
	
def cluster_object_percentile_default():
	return 0.90
	
def high_wcl_percentile_default():
	return 0.85 # Approximately 1-500/3366; see top of second column of p. 8 of the H&D paper'.
	
def structure_file_name_default():
	return ""
	

	
		
def hong_dey_run_config_file(configuration_file_name, configuration_section):

	import ConfigParser
	import sys

	defaults = {}
	defaults['filter_condition_scale'] = str(filter_condition_scale_default())
	defaults['graph_size_target_is_average_edges_per_vertex'] = str(graph_size_target_is_average_edges_per_vertex_default())
	defaults['graph_size_target'] = str(graph_size_target_default())
	defaults['alpha'] = str(alpha_default())
	defaults['filament_object_percentile'] = str(filament_object_percentile_default())
	defaults['cluster_object_percentile'] = str(cluster_object_percentile_default())
	defaults['high_wcl_percentile'] = str(high_wcl_percentile_default())
	defaults['structure_file_name'] = structure_file_name_default()

	if configuration_file_name[-4:] == ".cfg":
		config = ConfigParser.ConfigParser(defaults)
		config.read(configuration_file_name)
	else:
		raise "configuration_file_name " + configuration_file_name + " has unexpected type"

	hong_dey_run(
		catalogue_file_name = config.get(configuration_section, "catalogue_file_name"),
		structure_file_name = config.get(configuration_section, "structure_file_name"),
		id_field_name = config.get(configuration_section, "id_field_name"),
		position_specification = config.get(configuration_section, "position_specification"),
		position_field_names_str = config.get(configuration_section, "position_field_names_str"),
		filter_condition_str = config.get(configuration_section, "filter_condition_str"),
		filter_condition_scale = config.getfloat(configuration_section, "filter_condition_scale"),
		output_file_name = config.get(configuration_section, "output_file_name"),
		graph_size_target_is_average_edges_per_vertex = config.getboolean(configuration_section, "graph_size_target_is_average_edges_per_vertex"),
		graph_size_target = config.getfloat(configuration_section, "graph_size_target"),
		alpha = config.getfloat(configuration_section, "alpha"),
		filament_object_percentile = config.getfloat(configuration_section, "filament_object_percentile"),
		cluster_object_percentile = config.getfloat(configuration_section, "cluster_object_percentile"),
		high_wcl_percentile = config.getfloat(configuration_section, "high_wcl_percentile")
	)
	
	sys.exit()
		


def hong_dey_run(
	catalogue_file_name,
	structure_file_name,
	id_field_name,
	position_specification,
	position_field_names_str,
	filter_condition_str,
	filter_condition_scale,
	output_file_name,
	graph_size_target_is_average_edges_per_vertex,
	graph_size_target,
	alpha, #For alpha see p HD 122
	filament_object_percentile,
	cluster_object_percentile,
	high_wcl_percentile): 

	import sys	
	sys.path.append("/share/splinter/ucapwhi/cosmic_web/hong_dey/cpp")

	import numpy as np
	import pylab
	import os
	import astropy.io.fits as pyfits
	import glob
	import cosmic_web as cw
	import itertools
	import math
	import itertools
	import cPickle
	import time
	import cosmic_web_utilities as cwu
	import scipy.spatial as sp
	import scipy.optimize as so


	out_dict = {}
	start_time = time.time()

	#######################################
	print "Getting data"

	print "catalogue_file_name = " + str(catalogue_file_name)
	out_dict["catalogue_file_name"] = catalogue_file_name
	
	print "id_field_name = " + str(id_field_name)
	out_dict["id_field_name"] = id_field_name

	print "position_specification = " + str(position_specification)
	out_dict["position_specification"] = position_specification

	print "position_field_names_str = " + str(position_field_names_str)
	out_dict["position_field_names_str"] = position_field_names_str

	print "filter_condition_str = " + str(filter_condition_str)
	out_dict["filter_condition_str"] = filter_condition_str

	print "filter_condition_scale = " + str(filter_condition_scale)
	out_dict["filter_condition_scale"] = filter_condition_scale

	print "output_file_name = " + str(output_file_name)
	out_dict["output_file_name"] = output_file_name
	
	# Convert certain string inputs to arrays.
	position_field_names = cwu.string_array_from_string_representation(position_field_names_str)
	filter_condition = cwu.float_array_from_string_representation(filter_condition_str)

	filter_condition = scale_filter_condition(filter_condition, filter_condition_scale)
	
	print "filter_condition = " + str(filter_condition)
	out_dict["filter_condition"] = str(filter_condition)

	x = pyfits.open(catalogue_file_name)

	# Cast the data arrays from single-precision (i.e. type=float32) to double-precision (type=float=float64) as this is what the cosmic_web C++ module expects.
	# An alternative would be to amend cw so that it could handle arrays of different types; for this we might want to use PyArray_FROMANY.

	coord1_unfiltered = x[1].data.field(position_field_names[0]).astype(float,copy=False)
	coord2_unfiltered = x[1].data.field(position_field_names[1]).astype(float,copy=False)
	coord3_unfiltered = x[1].data.field(position_field_names[2]).astype(float,copy=False)
	id_unfiltered = (np.arange(coord1_unfiltered.shape[0]) if id_field_name == 'sequential' else x[1].data.field(id_field_name))

	x.close()

	(coord1, coord2, coord3, id) = cwu.triple_filter(coord1_unfiltered, coord2_unfiltered, coord3_unfiltered, filter_condition, id_unfiltered)
	
	if position_specification == "RA_DEC_REDSHIFT":
		out_dict["ra"] = coord1
		out_dict["dec"] = coord2
		out_dict["redshift"] = coord3
		(X, Y, Z) = cwu.spherical_to_cartesian(coord1, coord2, redshift = coord3)
	elif position_specification == "RA_DEC_DISTANCE":
		out_dict["ra"] = coord1
		out_dict["dec"] = coord2
		out_dict["distance"] = coord3
		(X, Y, Z) = cwu.spherical_to_cartesian(coord1, coord2, distance = coord3)
	elif position_specification == "X_Y_Z":
		(X, Y, Z) = (coord1, coord2, coord3)
	else:
		raise SystemError("Internal error - bad position_specification")
	
	
	out_dict["X"] = X
	out_dict["Y"] = Y
	out_dict["Z"] = Z
	out_dict["id"] = id


	# This section allows data to be plotted using gnuplot. Run gnuplot, and then within the interactive gnuplot session, give the command 'splot "gnuplot_input.txt"'.
	if False:
		np.savetxt('./output/gnuplot_input.txt', np.dstack((X,Y,Z))[0], delimiter='\t')

	num_vertices = len(X)
	print "num_vertices = " + str(num_vertices)
	out_dict["num_vertices"] = num_vertices
	
	if num_vertices == 0:
		raise SystemError("No galaxies found")


	# Build kdtree
	kd_tree = sp.cKDTree(np.column_stack((X, Y, Z)))

	# End of getting data


	


	#######################################
	# Start of determining the linking length. See the discussion in my notes HD 41-2. See also p. HD 117.
	print "Determine linking length"
			
	if graph_size_target_is_average_edges_per_vertex:
		if graph_size_target == 0.0:
			linking_length = 0.0
		else:
			fn_average_edges_per_vertex_error = lambda trial_linking_length: (kd_tree.count_neighbors(kd_tree, trial_linking_length) / float(num_vertices) - 1.0) - graph_size_target # See p. HD 139
			linking_length = so.bisect(fn_average_edges_per_vertex_error, 0.0, 50.0, disp=True)
	else:
		linking_length = graph_size_target


	print "linking length = " + str(linking_length)
	out_dict["linking_length"] = linking_length

	# End of determining the linking length.





	#######################################
	# Build graph
	print "Build graph"
	st = time.time()

	dc = np.zeros(num_vertices) # dc = num_edges_per_vertex; dc stands for 'degree centrality'
	wdc = np.zeros(num_vertices) # dc = weighted dc
	
	weighted_edge_graph_list = [] # See http://stackoverflow.com/questions/7133885/fastest-way-to-grow-a-numpy-numeric-array

	
	for (i,j) in kd_tree.query_pairs(linking_length):

		separation = np.sqrt((X[j] - X[i])**2 + (Y[j] - Y[i])**2 + (Z[j] - Z[i])**2)

		dc[i] += 1
		dc[j] += 1
				
		# Two different definitions of weight:
		weight_bc = np.maximum(separation/linking_length, 0.00001) # 0 for very close points, 1 for points separated by linking_length. Low cut off in case of two galaxies with zero separation.
                                                                                # Note that if we get here then certainly linking_length > 0.0.
		weight_dc = 2.0 - weight_bc                                # 2 for very close points, 1 for points separated by linking_length

		wdc[i] += weight_dc
		wdc[j] += weight_dc
				
		weighted_edge_graph_list.append(i)
		weighted_edge_graph_list.append(j)
		weighted_edge_graph_list.append(weight_bc)

				
	sum_dc = float(sum(dc)) # The sum counts each edge twice, once for each end.
	num_edges = sum_dc / 2.0
	print "num_edges = " + str(num_edges)
	out_dict["num_edges"] = num_edges
	average_edges_per_vertex = sum_dc / num_vertices 
	print "average_edges_per_vertex = " + str(average_edges_per_vertex)
	out_dict["average_edges_per_vertex"] = average_edges_per_vertex

	print "Elapsed time: " + str(time.time() - st)
	# End of building the graph.



	#######################################
	# 1. DC calculation (degree centrality); uses DC.
	print "DC calculation"
	out_dict["dc"] = dc
	# End of DC calculation


	if linking_length > 0.0:

		#######################################
		# 2. WBC calculation (weighted betweenness centrality); uses weighted BC over weighted DC.
		print "WBC calculation"
		st = time.time()

		weighted_edge_graph = np.reshape(weighted_edge_graph_list, newshape=(len(weighted_edge_graph_list)/3, 3))

		if True:
			print "\t Old wbc calculation"
			st1 = time.time()
			wbc = cw.centrality_calculation("betweenness_centrality", num_vertices, weighted_edge_graph)
			print "\t Elapsed time: " + str(time.time() - st1)
			out_dict["wbc"] = wbc
			if isinstance(wbc, basestring):
				raise SystemError("cw.centrality_calculation: " + wbc)

		
		if False:
			print "\t New wbc calculation"
			st1 = time.time()

			epsilon = 0.02
			out_dict["epsilon"] = epsilon
			delta = 0.1
			out_dict["delta"] = delta

			approx_wbc = cw.networkit_betweenness_centrality(num_vertices, weighted_edge_graph, epsilon, delta)
			print "\t Elapsed time: " + str(time.time() - st1)
			out_dict["approx_wbc"] = approx_wbc
			if isinstance(approx_wbc, basestring):
				raise SystemError("cw.networkit_betweenness_centrality: " + approx_wbc)

		out_dict["alpha"] = alpha
		wbc_over_wdc = np.multiply(wbc, np.power(np.maximum(wdc, 1.0), -alpha)) # If wdc is 0 then wbc will be as well; in this case set the ratio to 0.0. Note that wdc is at least 1.0.
		out_dict["wdc"] = wdc
		out_dict["wbc_over_wdc"] = wbc_over_wdc
		
		print "Elapsed time: " + str(time.time() - st)
		# End of WBC calculation


		#######################################
		# 3. Filament recognition
		# Get the subgraph of objects with high wbc_over_wdc; look at its connected components.
		if True:

			print "Filament recognition"

			
			out_dict["filament_object_percentile"] = filament_object_percentile

			wbc_over_wdc_classification = cwu.percentile_classification(wbc_over_wdc, [filament_object_percentile])
			filament_object_filter = np.where((wbc_over_wdc_classification==1))
			out_dict["filament_object_filter"] = filament_object_filter # T/F for each vertex to specify whether it is a filament object.

			filament_object_edge_graph_list = []

			filament_object_kd_tree = sp.cKDTree(np.column_stack((X[filament_object_filter], Y[filament_object_filter], Z[filament_object_filter])))
			filament_object_indices = (np.arange(num_vertices))[filament_object_filter] # can use this to relate indices in the filament object graph to indices in the main graph
			num_filament_objects = len(filament_object_indices)
			for (i,j) in filament_object_kd_tree.query_pairs(linking_length):
				filament_object_edge_graph_list.append(i)
				filament_object_edge_graph_list.append(j)
				filament_object_edge_graph_list.append(1.0)
			filament_object_edge_graph = np.reshape(filament_object_edge_graph_list, newshape=(len(filament_object_edge_graph_list)/3, 3))
		
			filament_identifier_for_filament_objects = cw.centrality_calculation("equivalence_classes", num_filament_objects, filament_object_edge_graph)
			# Now expand this to refer to all objects (using -1 to denote non-filamentary objects)
			filament_identifier = -1.0 * np.ones(num_vertices)
			for i in range(num_filament_objects):
				filament_identifier[filament_object_indices[i]] = filament_identifier_for_filament_objects[i]

			out_dict["filament_identifier"] = filament_identifier
			num_filaments = int(max(filament_identifier)) + 1
			out_dict["num_filaments"] = num_filaments

			# Now calculate a MST (minimum spanning tree) representation for each filament; also record the directions of the filament edges.
			MST_representation = []
			filament_directions = []
			for filament_index in range(num_filaments):
				filament_index_filter = np.where(filament_identifier == filament_index)
				X_filament_index = X[filament_index_filter]
				Y_filament_index = Y[filament_index_filter]
				Z_filament_index = Z[filament_index_filter]
				index_filament_index = (np.arange(num_vertices))[filament_index_filter] # This allows us to correspond between indices on the filament and indices in the entire catalogue.

				mst = cwu.minimum_spanning_tree_on_points(X_filament_index, Y_filament_index, Z_filament_index)
				for (i,j) in mst:
					MST_representation.append(filament_index)
					MST_representation.append(index_filament_index[i])
					MST_representation.append(index_filament_index[j])

					(dx, dy, dz) = (X_filament_index[j] - X_filament_index[i], Y_filament_index[j] - Y_filament_index[i], Z_filament_index[j] - Z_filament_index[i])
					filament_directions.append(dx if dz >= 0.0 else -dx)
					filament_directions.append(dy if dz >= 0.0 else -dy)
					filament_directions.append(dz if dz >= 0.0 else -dz)

			out_dict["filament_MST_indices"] = np.reshape(MST_representation, newshape=(len(MST_representation)/3, 3))
			out_dict["filament_directions"] = np.reshape(filament_directions, newshape=(len(filament_directions)/3, 3))

			

		#######################################
		# 4. Classification via 'distance from structure', to allow correlation of this quantity with local density. See p. HD 153.
		# Classification code: 3 = near red and blue; 2 = near red only; 1 = near blue only; 0 = near nothing
		if True:
			distance_from_structure_class = np.zeros(num_vertices)
			# High DC (red dots - potential cluster central members
			
			out_dict["cluster_object_percentile"] = cluster_object_percentile
			dc_classification = cwu.percentile_classification(dc, [cluster_object_percentile])
			for i in range(num_vertices):
				if dc_classification[i] == 1 or wbc_over_wdc_classification[i] == 1:
					# It's a red or blue dot
					for j in kd_tree.query_ball_point([X[i], Y[i], Z[i]], linking_length):
						# j is a neighbour of a red or blue dot
						if dc_classification[i] == 1 and distance_from_structure_class[j] < 2:
							distance_from_structure_class[j] += 2
						if wbc_over_wdc_classification[i] == 1 and distance_from_structure_class[j] % 2 == 0:
							distance_from_structure_class[j] += 1
			out_dict["distance_from_structure_class"] = distance_from_structure_class
							

		




		#######################################
		# 5. CL calculation (closeness centrality); uses weighted CL.
		if False:
			print "CL calculation"

			wcl = cw.centrality_calculation("closeness_centrality", num_vertices, weighted_edge_graph)
			out_dict["wcl"] = wcl

			wcl_two_set_classification = cwu.two_means_classification(wcl, 0.0)
		
			low_wcl_percentile = np.sum(np.where(wcl_two_set_classification == 0, 1.0, 0.0)) / float(num_vertices)
			#high_wcl_percentile = 0.85 # Approximately 1-500/3366; see top of second column of p. 8 of the H&D paper.

			out_dict["low_wcl_percentile"] = low_wcl_percentile
			out_dict["high_wcl_percentile"] = high_wcl_percentile


			wcl_classification = cwu.percentile_classification(wcl, np.array((low_wcl_percentile, high_wcl_percentile)))

			out_dict["wcl_classification"] = wcl_classification

			# It is helpful to see the equivalence classes (as this aids the analysis of the wcl classification results - see my notes p. HD55).
			equivalence_classes = cw.centrality_calculation("equivalence_classes", num_vertices, weighted_edge_graph)
			out_dict["equivalence_classes"] = equivalence_classes

			# End of CL calculation




		#######################################
		# 6. Dimensionality

		# See p. HD 158
		if True:
			print "Dimensionality"
			# For now, pick a point at random
			i = int(num_vertices * np.random.rand())
			dimensionality_num_points  = 101
			dimensionality_x = np.linspace(0.0, 10.0 * linking_length, dimensionality_num_points)
			dimensionality_y = np.zeros(dimensionality_num_points)
			for k in range(dimensionality_num_points):
				dimensionality_y[k] = len(kd_tree.query_ball_point([X[i], Y[i], Z[i]], dimensionality_x[k]))
			out_dict["dimensionality_x"] = dimensionality_x
			out_dict["dimensionality_y"] = dimensionality_y

		#######################################
		# 7. Comparison of results with structure file
		# For toy models we have a 'structure file' speciying the locations of the clusters and filaments.
		# The code that follows assesses the correspondence between the HD output and the structures.
		if structure_file_name != "":
			with open(structure_file_name, "rb") as ofile:
				structures = cPickle.load(ofile)
			for structure in structures:
				if structure["object"] == "filament":
					cylinder_endpoint = [np.array(cwu.get_structure_by_name(structures, name)["centre"]) for name in structure["location"].split("-")] # A two-element list
					dist_between_cylinder_endpoints_using_filaments = sys.float_info.max
					for filament_index in range(num_filaments):
						filament_index_filter = np.where(filament_identifier == filament_index)
						X_filament_index = X[filament_index_filter]
						Y_filament_index = Y[filament_index_filter]
						Z_filament_index = Z[filament_index_filter]
						dist_between_cylinder_endpoints_using_this_filament = sum([np.amin(np.sqrt((cylinder_endpoint[i][0] - X_filament_index)**2 + (cylinder_endpoint[i][1] - Y_filament_index)**2 + (cylinder_endpoint[i][2] - Z_filament_index)**2)) for i in range(2)]) # See p. HD 469
						dist_between_cylinder_endpoints_using_filaments = min(dist_between_cylinder_endpoints_using_filaments, dist_between_cylinder_endpoints_using_this_filament)
					out_dict[structure["name"]+ "_dist"] = dist_between_cylinder_endpoints_using_filaments
							
							
						
						
						
		

	#######################################
	# 8. Put all the results together
	print "Output"

	elapsed_time = time.time() - start_time # Have to put this here so it can be written to file.
	with open(output_file_name, "wb") as output_file:
		cPickle.dump(out_dict, output_file)


	print "Total elapsed time = " + str(elapsed_time)





def standard_z_means():
	return [0.24, 0.27, 0.30, 0.36, 0.465, 0.48, 0.57, 0.66]

	
def hong_dey_versus_linking_length():
	for i in range(1, 20):
		hong_dey_run_configuration(configuration = 1, linking_length = float(i)/2.0)


def create_graphs_for_20160427_call():
	for use_z_spec in [True, False]:
		for z_mean in standard_z_means():
			print use_z_spec, z_mean
			hong_dey_run_configuration(configuration = 6, target_average_edges_per_vertex = 6.0, z_mean = z_mean, use_z_spec = use_z_spec)

def create_graphs_for_20160509_conference():
	for use_z_spec in [False]:
		for z_mean in standard_z_means():
			print use_z_spec, z_mean
			hong_dey_run_configuration(configuration = 8, target_average_edges_per_vertex = 6.0, z_mean = z_mean, use_z_spec = use_z_spec)

		
#### MAIN PROCEDURE STARTS HERE	
	
try:
	import sys
	
	#hong_dey_run_config_file(sys.argv[1], sys.argv[2])

	hong_dey_run_configuration(configuration = 17, target_average_edges_per_vertex = 6.0)
	#hong_dey_run_configuration(configuration = 15, target_average_edges_per_vertex = float(sys.argv[1]))
	#hong_dey_run_configuration(configuration = 8, target_average_edges_per_vertex = 6.0, use_z_spec = False, z_mean = standard_z_means()[int(sys.argv[1])])
	#hong_dey_run_configuration(configuration = 12, target_average_edges_per_vertex = 6.0, run_num = int(sys.argv[1]))

	




except SystemError as e:
	errMsg = e.args[0]
	print errMsg



