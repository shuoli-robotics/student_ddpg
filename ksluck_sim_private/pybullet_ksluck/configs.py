ast_6legged_config = {
    # Robot params
    'robot' : '6legged',
    'initial_height' : 0.001,
    'self_collision' : False,
    'control_mode' : 'velocity',
    'joint_limit_motor_1' : (-0.35, 0.35),
    'joint_limit_motor_2' : (0, 0.8),
    'joint_limit_motor_3' : (0, 2.0),
    #Simulator params
    'restitution' : 0.1,
    'lateralFriction' : 0.85,
    'action_repeat' : 8,
    'render' : False,
    'sim_timestep' : 0.0165,
    'sim_frameskip' : 4,
    'sim_numSolverIterations' : 10,
    'do_hard_reset' : False,
    'COV_ENABLE_PLANAR_REFLECTION_plane' : 0,
    'max_nmbr_of_time_steps': 600,
    'COV_ENABLE_SHADOWS': False
}

fast_standing_6legged_config = {
    # Robot params
    'robot' : '6legged',
    'initial_height' : 0.01 + 0.3 - 0.15, #0.112,
    'self_collision' : False,
    'control_mode' : 'velocity',
    'joint_limit_motor_1' : (-0.35, 0.35),
    'joint_limit_motor_2' : (0, 0.8),
    'joint_limit_motor_3' : (0, 2.0),
    #Simulator params
    'restitution' : 0.1,
    'lateralFriction' : 0.85,
    'action_repeat' : 4,
    'render' : False,
    'sim_timestep' : 0.0165,
    'sim_frameskip' : 4,
    'sim_numSolverIterations' : 10,
    'do_hard_reset' : False,
    'COV_ENABLE_PLANAR_REFLECTION_plane' : 0,
    'max_nmbr_of_time_steps': 600,
    'COV_ENABLE_SHADOWS': False,
    'initial_joint_positions' : {
        'L_F_motor_3/X8_9': -1.57,
        'L_M_motor_3/X8_9': -1.57,
        'L_B_motor_3/X8_9': -1.57,
        'R_F_motor_3/X8_9': 1.57,
        'R_M_motor_3/X8_9': 1.57,
        'R_B_motor_3/X8_9': 1.57},
}
