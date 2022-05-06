import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import ros_numpy
import numpy as np
import time
import math

import tensorflow as tf
from tf_agents.trajectories import time_step as ts
import tf_agents.trajectories as tr
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.environments import tf_py_environment

from Discrete_Target_Search import DiscreteTargetSearchEnv

input_size = 43;

policy_dir = './final_policy'
depth_topic = '/locobot/camera/depth/color/points'
odom_topic = '/locobot/odom'
velocity_topic = '/locobot/mobile_base/commands/velocity'

def pointsHandler(msg):

	global input_size, obs
	global robot_x, robot_y, robot_theta, target_x, target_y

	pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=False)

	dist_limit = 10.0
	y_limit = -0.1

	obstacles = np.zeros(input_size)
	dists = np.zeros(input_size)

	for i in range(0, pc.shape[0], 2):
		for j in range(0, pc.shape[1], 2):
			x = pc[i][j][0]
			y = pc[i][j][1]
			z = pc[i][j][2]

			

			if y < y_limit:
				input_idx = round((j / (msg.width-1)) * (input_size-1))
				dists[input_idx] = z
				if z < dist_limit:
					obstacles[input_idx] = 1

	print('[', end='')
	for i in range(input_size):
		if obstacles[i] == 1:
			print('X', end='')
		else:
			print(' ', end='')
	print(']')
	
	#model_input = list(np.array(list(zip(obstacles, dists)), dtype=np.float32).reshape(1, input_size*2))
	model_input = []
	for i in range(input_size):
		model_input.append(obstacles[i])
	for i in range(input_size):
		model_input.append(dists[i])
	model_input.append(robot_x)
	model_input.append(robot_y)
	model_input.append(robot_theta)
	model_input.append(target_x)
	model_input.append(target_y)

	obs = model_input


def odomHandler(msg):
	global robot_x, robot_y, robot_theta
	if msg.child_frame_id == 'base_footprint':
		robot_x = msg.pose.pose.position.x
		robot_y = msg.pose.pose.position.y

		o = msg.pose.pose.orientation
		w = o.w
		x = o.x
		y = o.y
		z = o.z
		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		robot_theta = math.atan2(t3, t4)
		#print("x: ", robot_x)
		#print("y: ", robot_y)
		#print("theta: ", robot_theta)
		#print('')

def turn(theta):
	global vel_pub, robot_theta

	cur_theta = robot_theta
	target_theta = cur_theta + theta
	if target_theta > math.pi:
		target_theta = -2*math.pi + target_theta
	elif target_theta < -math.pi:
		target_theta = 2*math.pi + target_theta

	speed = 1.0

	msg = Twist()
	if theta > 0:	
		msg.angular.z = speed
	else:
		msg.angular.z = -speed

	vel_pub.publish(msg)
	print("turning")

	while abs(robot_theta-target_theta) > 0.08:
		pass
	
	msg.angular.z = 0.0
	vel_pub.publish(msg)

def go_forward(x):
	global vel_pub

	speed = 1.0

	msg = Twist()
	msg.linear.x = speed
	vel_pub.publish(msg)
	time.sleep(x / speed)
	msg.linear.x = 0.0
	vel_pub.publish(msg)

def turn_right():
	global vel_pub

	msg = Twist()
	msg.angular.z = -0.3
	vel_pub.publish(msg)
	time.sleep(1)
	msg.angular.z = 0.0
	vel_pub.publish(msg)

def get_action(obs):
	global runenv
	global env
	global saved_policy
	global policy_state

	runenv.obs = obs
	time_step = env.reset()
	policy_step = saved_policy.action(time_step, policy_state)
	policy_state = policy_step.state

	return int(policy_step.action)

saved_policy = tf.compat.v2.saved_model.load(policy_dir)
policy_state = saved_policy.get_initial_state(batch_size=1)

obs = [ 0.0 for x in range(43*2+5) ]
runenv = DiscreteTargetSearchEnv()
env = tf_py_environment.TFPyEnvironment(runenv)
runenv.obs = obs
time_step = env.reset()
policy_step = saved_policy.action(time_step, policy_state)
policy_state = policy_step.state

robot_x = 0
robot_y = 0
robot_theta = 0

target_x = 4
target_y = 1


rospy.init_node('robot_controller')

depth_sub = rospy.Subscriber(depth_topic, PointCloud2, pointsHandler, queue_size=1, buff_size=2**24)
odom_sub = rospy.Subscriber(odom_topic, Odometry, odomHandler, queue_size=1, buff_size=2**24)
vel_pub = rospy.Publisher(velocity_topic, Twist)

while not rospy.is_shutdown():
	action = get_action(obs)
	print(action)

	if action == 0:
		#pass
		turn(-1)
		go_forward(0.1)
	elif action == 1:
		#pass
		turn(-0.5)
		go_forward(0.5)
	elif action == 2:
		#pass
		go_forward(1)
	elif action == 3:
		turn(0.5)
		go_forward(0.5)
	elif action == 4:
		turn(1)
		go_forward(0.1)

	#time.sleep(1)


exit