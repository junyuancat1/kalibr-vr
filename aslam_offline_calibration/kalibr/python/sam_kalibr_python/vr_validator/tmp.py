import matplotlib.pyplot as plt
import numpy as np
import cv2

def E_from_a_2_b(camera_a_Rbc, camera_a_tbc, camera_b_Rbc, camera_b_tbc, camera_a_Rcw, camera_a_tcw):

    R_camera_a_to_camera_b = np.dot(camera_b_Rbc.T, camera_a_Rbc)
    T_camera_a_to_camera_b = np.dot(camera_b_Rbc.T, camera_a_tbc - camera_b_tbc)
    R_camera_b_to_world = np.dot(R_camera_a_to_camera_b, camera_a_Rcw)
    T_camera_b_to_world = np.dot(R_camera_a_to_camera_b, camera_a_tcw).reshape(1,3) + T_camera_a_to_camera_b
    r_camera_b_to_world, _ = cv2.Rodrigues(R_camera_b_to_world)

    return r_camera_b_to_world, T_camera_b_to_world

def plot_3d_line(ax, p1, p2, color=None, linestyle='-', linewidth=1.0):
    if color is None:
        color = np.random.rand(3)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], linestyle=linestyle, linewidth=linewidth, color=color)
    
# plot line [0,0,0] -> [1,0,0] in red
# plot line [0,0,0] -> [0,1,0] in green
# plot line [0,0,0] -> [0,0,1] in blue

def plot_axis(ax,Rwc,twc):
    # plot line [0,0,0] -> [1,0,0] in red
    # plot line [0,0,0] -> [0,1,0] in green
    # plot line [0,0,0] -> [0,0,1] in blue
    plot_3d_line(ax, twc, twc+Rwc[:,0], color='r')
    plot_3d_line(ax, twc, twc+Rwc[:,1], color='g')
    plot_3d_line(ax, twc, twc+Rwc[:,2], color='b')
    
    

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 计算在坐标系1下，坐标系2的位姿
# E2 · Pw = Pc2
# R2 · Pw + t2 = Pc2 = (0,0,0)
# R2 · Pw = -t2
# Pw = R2.T · (-t2)

vr_camera1_Rbc1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
vr_camera1_tbc1 = np.array([0, 0, 0])

vr_camera2_Rbc2 = np.array([[-0.99949765, -0.025876258, 0.018299539], [0.029883759, -0.96176534, 0.27223959], [0.010555321, 0.27264969, 0.96205547]]).reshape(3,3)
vr_camera2_tbc2 = np.array([-0.00038707237, 0.10318894, -0.01401102]) 
vr_camera2_rbc2, _ = cv2.Rodrigues(vr_camera2_Rbc2)
Pc2w2 = np.dot(vr_camera2_Rbc2.T, -vr_camera2_tbc2) 

vr_camera3_Rbc3 = np.array([[0.5687224, 0.81622218, -0.10166702], [-0.3726157, 0.36585493, 0.85282338], [0.73328874, -0.44713703, 0.51220709]]).reshape(3,3)
vr_camera3_tbc3 = np.array([-0.0056250621, 0.041670205, -0.013388009]) 
vr_camera3_rbc3, _ = cv2.Rodrigues(vr_camera3_Rbc3)
Pc3w3 = np.dot(vr_camera3_Rbc3.T, -vr_camera3_tbc3)

vr_camera4_Rbc4 = np.array([[-0.59714752, 0.79234879, -0.12489289], [-0.35174627, -0.11873178, 0.92853504], [0.72089486, 0.598403, 0.34960612]]).reshape(3,3)
vr_camera4_tbc4 = np.array([-0.079096205, 0.064828795, -0.066688745])
vr_camera4_rbc4, _ = cv2.Rodrigues(vr_camera4_Rbc4)
Pc4w4 = np.dot(vr_camera4_Rbc4.T, -vr_camera4_tbc4)

print(Pc2w2)
print(Pc3w3)
print(Pc4w4)

# exit(0)


print(vr_camera2_Rbc2)
print(vr_camera2_tbc2)


fresh_rc1w = np.float32([-1.31864372, 2.66745841, -0.45909558]).reshape(1,3)
fresh_tc1w = np.float32([0.21389172, 0.11939645, 0.27458963]).reshape(3,1)
fresh_Rc1w, _ = cv2.Rodrigues(fresh_rc1w)

fresh_rc2w = np.float32([-2.78232977, -1.2651627, 0.40838129]).reshape(1,3)
fresh_tc2w = np.float32([-0.21116153, 0.07293855, 0.28693892]).reshape(3,1)
fresh_Rc2w, _ = cv2.Rodrigues(fresh_rc2w)

fresh_rc3w = np.float32([0.01867105, 2.37005514, -1.92390398]).reshape(1,3)
fresh_tc3w = np.float32([0.19028125, 0.23620335, 0.22807506]).reshape(3,1)
fresh_Rc3w, _ = cv2.Rodrigues(fresh_rc3w)

fresh_rc4w = np.float32([1.40496441, 1.60617537, -1.39236939]).reshape(1,3)
fresh_tc4w = np.float32([-0.14519866, 0.22861433, 0.25446183]).reshape(3,1)
fresh_Rc4w, _ = cv2.Rodrigues(fresh_rc4w)

rc2w_1, tc2w_1 = E_from_a_2_b(vr_camera1_Rbc1, vr_camera1_tbc1, vr_camera2_Rbc2, vr_camera2_tbc2, fresh_Rc1w, fresh_tc1w)
print("tc2w_1: ", tc2w_1 , "====>", fresh_tc2w.T)
print("rc2w_1: ", rc2w_1.T , "====>", fresh_rc2w)

rc3w_1, tc3w_1 = E_from_a_2_b(vr_camera1_Rbc1, vr_camera1_tbc1, vr_camera3_Rbc3, vr_camera3_tbc3, fresh_Rc1w, fresh_tc1w)
print("tc3w_1: ", tc3w_1 , "====>", fresh_tc3w.T)
print("rc3w_1: ", rc3w_1.T , "====>", fresh_rc3w)

rc4w_1, tc4w_1 = E_from_a_2_b(vr_camera1_Rbc1, vr_camera1_tbc1, vr_camera4_Rbc4, vr_camera4_tbc4, fresh_Rc1w, fresh_tc1w)
print("tc4w_1: ", tc4w_1 , "====>", fresh_tc4w.T)
print("rc4w_1: ", rc4w_1.T , "====>", fresh_rc4w)


exit(0)
# plot_axis(ax, vr_camera1_Rbc1, vr_camera1_tbc1 * 100)
# plot_axis(ax, vr_camera2_Rbc2, vr_camera2_tbc2 * 100)
# plot_axis(ax, vr_camera3_Rbc3, vr_camera3_tbc3 * 100)
# plot_axis(ax, vr_camera4_Rbc4, vr_camera4_tbc4 * 100)

plot_axis(ax, fresh_Rc1w, fresh_tc1w * 100) # trackingA
plot_axis(ax, fresh_Rc2w, fresh_tc2w * 100) # trackingB
plot_axis(ax, fresh_Rc3w, fresh_tc3w * 100) # ctltrackingA
plot_axis(ax, fresh_Rc4w, fresh_tc4w * 100) # ctltrackingB

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()



