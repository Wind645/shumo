import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.font_manager as fm

# Force enable 3D plotting with proper registration
HAS_3D = True
try:
    from mpl_toolkits.mplot3d import Axes3D
    import mpl_toolkits.mplot3d.proj3d as proj3d
    # Force register 3D projection
    from mpl_toolkits.mplot3d import proj3d
    from matplotlib.projections import register_projection
    print("3D plotting successfully loaded and registered.")
except Exception as e:
    print(f"Warning: {e}")
    print("Attempting alternative 3D setup...")
    HAS_3D = False

def setup_mixed_font():
    """设置中英文混合字体"""
    try:
        # 使用AR PL UMing字体支持中文
        font_path = '/usr/share/fonts/truetype/arphic/uming.ttc'
        font_prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False
    except:
        # 备用字体设置
        try:
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

setup_mixed_font()

def create_dynamic_top_view(question2: bool=False):
    """创建动态俯视图 (question2=True 时仅保留真假目标、M1、FY1)"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 目标点设置
    fake_target = np.array([0, 0])
    real_target = np.array([0, 200])
    
    # 导弹数据
    missiles_data = {
        'M1': {'pos': np.array([20000, 0]), 'color': 'orange'},
        'M2': {'pos': np.array([19000, 600]), 'color': 'orangered'},
        'M3': {'pos': np.array([18000, -600]), 'color': 'red'}
    }
    # 无人机数据
    drones_config = {
        'FY1': {
            'pos': np.array([17800, 0]), 
            'color': 'green',
            'direction': np.array([-1, 0]),
            'speed': 120,
            'bombs': [
                {'deploy_time': 1.5, 'explode_delay': 3.6},
                {'deploy_time': 5.0, 'explode_delay': 3.6},
                {'deploy_time': 8.5, 'explode_delay': 3.6}
            ]
        },
        'FY2': {'pos': np.array([12000, 1400]), 'color': 'lime','direction': np.array([-0.8, -0.6]),'speed': 110,'bombs': [{'deploy_time': 2.0,'explode_delay': 3.5},{'deploy_time': 6.0,'explode_delay': 3.7},{'deploy_time': 10.0,'explode_delay': 3.9}]},
        'FY3': {'pos': np.array([6000, -3000]), 'color': 'forestgreen','direction': np.array([-0.6, 0.8]),'speed': 100,'bombs': [{'deploy_time': 1.8,'explode_delay': 3.3},{'deploy_time': 5.5,'explode_delay': 3.6},{'deploy_time': 9.2,'explode_delay': 4.0}]},
        'FY4': {'pos': np.array([11000, 2000]), 'color': 'darkgreen','direction': np.array([-0.9, -0.4]),'speed': 130,'bombs': [{'deploy_time': 2.2,'explode_delay': 3.4},{'deploy_time': 6.5,'explode_delay': 3.8},{'deploy_time': 10.8,'explode_delay': 3.5}]},
        'FY5': {'pos': np.array([13000, -2000]), 'color': 'lightgreen','direction': np.array([-0.7, 0.7]),'speed': 125,'bombs': [{'deploy_time': 1.2,'explode_delay': 3.6},{'deploy_time': 4.8,'explode_delay': 3.3},{'deploy_time': 8.3,'explode_delay': 3.7}]}
    }
    # 第二题模式过滤
    if question2:
        missiles_data = {'M1': missiles_data['M1']}
        drones_config = {'FY1': drones_config['FY1']}
        # 第二题模式：只保留一枚烟幕弹
        if len(drones_config['FY1']['bombs']) > 1:
            drones_config['FY1']['bombs'] = [drones_config['FY1']['bombs'][0]]
    
    # 目标点设置（根据题目要求修正）
    real_target = np.array([0, 200])  # 真目标圆心位置 (0, 200, 0)
    
    # 仿真参数（根据题目要求）
    missile_speed = 300  # m/s（题目明确指定）
    smoke_drop_speed = 3  # m/s（题目明确指定烟幕云团下沉速度）
    smoke_effective_radius = 10  # m（题目指定中心10m范围内有效）
    smoke_effective_time = 20    # s（题目指定起爆20s内有效遮蔽）
    dt = 0.1            # 时间步长(s)
    total_time = 80     # 总仿真时间(s)
    
    # 为每个导弹计算到假目标的方向向量
    for name, data in missiles_data.items():
        direction = fake_target - data['pos']
        distance = np.linalg.norm(direction)
        data['direction'] = direction / distance
        data['flight_time'] = distance / missile_speed
    
    # 存储轨迹和烟幕弹
    missile_trajectories = {name: [data['pos'].copy()] for name, data in missiles_data.items()}
    drone_trajectories = {name: [config['pos'].copy()] for name, config in drones_config.items()}
    smoke_bombs = []  # 烟幕弹列表
    deployed_bombs = {name: [] for name in drones_config.keys()}  # 记录已投放的烟幕
    
    def animate(frame):
        ax.clear()
        current_time = frame * dt
        
        # 设置坐标轴
        ax.set_xlim(-2000, 22000)
        ax.set_ylim(-4000, 4000)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'A题烟幕干扰弹投放策略仿真 - 时间: {current_time:.1f}s' + (' (第二题模式)' if question2 else ''), fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 绘制假目标（原点）
        ax.scatter(*fake_target, color='red', s=300, marker='*', label='假目标(原点)', zorder=5, 
                  edgecolors='darkred', linewidth=2)
        
        # 绘制真目标（圆柱形，半径7m，高10m）
        real_target_circle = plt.Circle(real_target, 7, color='blue', fill=False, linewidth=3, label='真目标(圆柱形)', alpha=0.8)
        ax.add_patch(real_target_circle)
        ax.scatter(*real_target, color='blue', s=200, marker='s', zorder=5)
        
        # 添加真目标标注
        ax.annotate('真目标\n半径7m 高10m', xy=real_target, xytext=(real_target[0]+500, real_target[1]+500),
                   arrowprops=dict(arrowstyle='->', color='blue'), fontsize=10, color='blue')
        
        # 更新导弹位置
        for name, data in missiles_data.items():
            if current_time <= data['flight_time']:
                new_pos = data['pos'] + data['direction'] * missile_speed * current_time
                missile_trajectories[name].append(new_pos.copy())
                
                ax.scatter(*new_pos, color=data['color'], s=200, marker='^', 
                          label=f'导弹{name}(300m/s)', zorder=4, edgecolors='black', linewidth=1)
                # 添加导弹编号标注
                ax.text(new_pos[0]+120, new_pos[1]+120, name, color=data['color'], fontsize=9,
                        ha='left', va='bottom', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none'))
                
                if len(missile_trajectories[name]) > 1:
                    traj = np.array(missile_trajectories[name])
                    ax.plot(traj[:, 0], traj[:, 1], color=data['color'], alpha=0.7, linewidth=2)
                
                # 显示导弹飞向假目标的方向
                ax.arrow(new_pos[0], new_pos[1], fake_target[0]-new_pos[0], fake_target[1]-new_pos[1],
                        head_width=150, head_length=200, fc=data['color'], ec=data['color'], 
                        alpha=0.4, length_includes_head=True, linestyle='--')
        
        # 更新无人机位置（等高度匀速直线飞行）
        for drone_name, config in drones_config.items():
            # 无人机按固定方向和速度飞行
            new_pos = config['pos'] + config['direction'] * config['speed'] * current_time
            drone_trajectories[drone_name].append(new_pos.copy())
            
            ax.scatter(*new_pos, color=config['color'], s=150, marker='o', 
                      label=f'无人机{drone_name}({config["speed"]}m/s)', zorder=4, edgecolors='black', linewidth=1)
            # 添加无人机编号标注
            ax.text(new_pos[0]+120, new_pos[1]-120, drone_name, color=config['color'], fontsize=9,
                    ha='left', va='top', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none'))
            
            # 绘制无人机轨迹
            if len(drone_trajectories[drone_name]) > 1:
                traj = np.array(drone_trajectories[drone_name])
                ax.plot(traj[:, 0], traj[:, 1], color=config['color'], 
                       alpha=0.7, linewidth=2, linestyle=':')
            
            # 检查是否需要投放烟幕弹（间隔至少1秒）
            for i, bomb in enumerate(config['bombs']):
                bomb_id = f"{drone_name}_{i}"
                if (current_time >= bomb['deploy_time'] and 
                    current_time < bomb['deploy_time'] + dt and
                    bomb_id not in deployed_bombs[drone_name]):
                    
                    # 在当前位置投放烟幕弹
                    bomb_pos = new_pos.copy()
                    smoke_bombs.append({
                        'pos': bomb_pos.copy(),
                        'init_pos': bomb_pos.copy(),              # 记录初始位置
                        'h_vel': config['direction'][:2] * config['speed'],  # 水平速度(与无人机一致)
                        'deploy_time': current_time,
                        'exploded': False,
                        'explode_time': current_time + bomb['explode_delay'],
                        'drone': drone_name,
                        'bomb_id': bomb_id
                    })
                    deployed_bombs[drone_name].append(bomb_id)
        
        # 更新烟幕弹状态
        active_smoke_count = 0
        for bomb in smoke_bombs:
            bomb_time = current_time - bomb['deploy_time']
            
            if not bomb['exploded'] and current_time >= bomb['explode_time']:
                bomb['exploded'] = True
                bomb['smoke_center'] = bomb['pos'].copy()  # 取爆炸瞬间更新后的位置
            
            if bomb['exploded']:
                explosion_time = current_time - bomb['explode_time']
                if explosion_time <= smoke_effective_time:  # 有效遮蔽时间20秒
                    smoke_center = bomb['smoke_center'].copy()
                    # 烟幕云团以3m/s速度下沉
                    smoke_center[1] -= smoke_drop_speed * explosion_time  # 在2D视图中用Y坐标表示下沉
                    active_smoke_count += 1
                    
                    # 绘制烟幕圆圈（有效范围10m）
                    smoke_circle = plt.Circle(smoke_center, smoke_effective_radius, color='gray', alpha=0.5)
                    ax.add_patch(smoke_circle)
                    ax.scatter(*smoke_center, color='gray', s=100, marker='*', zorder=4)
            else:
                # 未爆炸：抛物线运动 (水平匀速，当前无垂直轴显示，直接水平位移)
                if bomb_time >= 0:
                    bomb['pos'][0] = bomb['init_pos'][0] + bomb['h_vel'][0] * bomb_time
                    bomb['pos'][1] = bomb['init_pos'][1] + bomb['h_vel'][1] * bomb_time
                    ax.scatter(*bomb['pos'], color='black', s=80, marker='o', zorder=4)
        
        # 添加详细信息
        total_deployed = len(smoke_bombs)
        info_text = f"""仿真状态:{' (第二题模式)' if question2 else ''}
时间: {current_time:.1f}s
已投放烟幕弹: {total_deployed}枚
活跃烟幕团: {active_smoke_count}个
导弹速度: 300m/s
烟幕下沉速度: 3m/s
有效遮蔽半径: 10m
有效遮蔽时间: 20s"""
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        # 添加图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if active_smoke_count > 0:
            by_label['烟幕云团(10m半径)'] = plt.Circle((0, 0), 1, color='gray', alpha=0.5)
        if len(smoke_bombs) > 0:
            by_label['烟幕弹'] = ax.scatter([], [], color='black', s=80, marker='o')
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 创建动画
    frames = int(total_time / dt)
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=200, repeat=True)
    
    return fig, ani

def create_dynamic_simulation(question2: bool=False):
    """创建3D动态仿真 - question2=True 时仅保留真假目标、M1、FY1"""
    print("强制启动3D模式..." + (" (第二题模式)" if question2 else ""))
    
    try:
        # Alternative 3D setup method
        import mpl_toolkits.mplot3d
        from mpl_toolkits.mplot3d import Axes3D
        
        # Force create 3D figure
        fig = plt.figure(figsize=(16, 12))
        
        # Try different ways to create 3D subplot
        try:
            ax = fig.add_subplot(111, projection='3d')
        except:
            print("尝试备用3D创建方法...")
            ax = Axes3D(fig)
            fig.add_axes(ax)
        
        print("3D坐标轴创建成功!")
        
    except Exception as e:
        print(f"3D创建失败，使用伪3D模式: {e}")
        return create_pseudo_3d_simulation()
    
    # 设定坐标系范围
    ax.set_xlim(-2000, 22000)
    ax.set_ylim(-4000, 4000)
    ax.set_zlim(0, 2500)
    
    # 设置初始视角（用户可以手动调整）
    ax.view_init(elev=25, azim=45)
    
    # 目标点设置（根据题目要求）
    fake_target = np.array([0, 0, 0])  # 假目标为原点
    real_target_center = np.array([0, 200, 0])  # 真目标下底面圆心 (0, 200, 0)
    
    # 导弹初始位置和目标（根据题目精确设置）
    missiles_data = {
        'M1': {'pos': np.array([20000, 0, 2000]), 'color': 'orange'},
        'M2': {'pos': np.array([19000, 600, 2100]), 'color': 'orangered'},
        'M3': {'pos': np.array([18000, -600, 1900]), 'color': 'red'}
    }
    
    # 无人机初始位置和飞行参数配置（根据题目精确设置）
    drones_config = {
        'FY1': {'pos': np.array([17800, 0, 1800]), 'color': 'green','direction': np.array([-1, 0, 0]),'speed': 120,'bombs': [ {'deploy_time': 1.5,'explode_delay': 3.6},{'deploy_time': 5.0,'explode_delay': 3.6},{'deploy_time': 8.5,'explode_delay': 3.6} ]},
        'FY2': {'pos': np.array([12000, 1400, 1400]), 'color': 'lime','direction': np.array([-0.8, -0.6, 0]),'speed': 110,'bombs': [{'deploy_time': 2.0,'explode_delay': 3.5},{'deploy_time': 6.0,'explode_delay': 3.7},{'deploy_time': 10.0,'explode_delay': 3.9}]},
        'FY3': {'pos': np.array([6000, -3000, 700]), 'color': 'forestgreen','direction': np.array([-0.6, 0.8, 0]),'speed': 100,'bombs': [{'deploy_time': 1.8,'explode_delay': 3.3},{'deploy_time': 5.5,'explode_delay': 3.6},{'deploy_time': 9.2,'explode_delay': 4.0}]},
        'FY4': {'pos': np.array([11000, 2000, 1800]), 'color': 'darkgreen','direction': np.array([-0.9, -0.4, 0]),'speed': 130,'bombs': [{'deploy_time': 2.2,'explode_delay': 3.4},{'deploy_time': 6.5,'explode_delay': 3.8},{'deploy_time': 10.8,'explode_delay': 3.5}]},
        'FY5': {'pos': np.array([13000, -2000, 1300]), 'color': 'lightgreen','direction': np.array([-0.7, 0.7, 0]),'speed': 125,'bombs': [{'deploy_time': 1.2,'explode_delay': 3.6},{'deploy_time': 4.8,'explode_delay': 3.3},{'deploy_time': 8.3,'explode_delay': 3.7}]}
    }
    if question2:
        missiles_data = {'M1': missiles_data['M1']}
        drones_config = {'FY1': drones_config['FY1']}
        # 第二题模式：只保留一枚烟幕弹
        if len(drones_config['FY1']['bombs']) > 1:
            drones_config['FY1']['bombs'] = [drones_config['FY1']['bombs'][0]]
    # 仿真参数（根据题目要求）
    missile_speed = 300  # m/s
    smoke_drop_speed = 3 # m/s (烟幕下沉速度)
    smoke_effective_radius = 10  # m
    smoke_effective_time = 20    # s
    dt = 0.1            # 时间步长(s)
    total_time = 80     # 总仿真时间(s)
    
    # 为每个导弹计算到假目标的方向向量
    for name, data in missiles_data.items():
        direction = fake_target - data['pos']
        distance = np.linalg.norm(direction)
        data['direction'] = direction / distance
        data['flight_time'] = distance / missile_speed
    
    # 存储轨迹
    missile_trajectories = {name: [data['pos'].copy()] for name, data in missiles_data.items()}
    drone_trajectories = {name: [config['pos'].copy()] for name, config in drones_config.items()}
    smoke_bombs = []  # 烟幕弹列表
    deployed_bombs = {name: [] for name in drones_config.keys()}  # 记录已投放的烟幕弹
    
    def animate(frame):
        try:
            ax.clear()
            current_time = frame * dt
            
            print(f"渲染帧 {frame}, 时间: {current_time:.1f}s")
            
            # 设置坐标轴
            ax.set_xlim(-2000, 22000)
            ax.set_ylim(-4000, 4000)
            ax.set_zlim(0, 2500)
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_zlabel('Z (m)', fontsize=12)
            ax.set_title(f'A题烟幕干扰弹投放策略3D仿真 - 时间: {current_time:.1f}s' + (' (第二题模式)' if question2 else ''), fontsize=14, fontweight='bold')
            
            # 绘制假目标（原点）
            ax.scatter(*fake_target, color='red', s=300, marker='*', label='假目标(原点)', edgecolors='darkred', linewidth=2)
            
            # 绘制真目标（圆柱形：半径7m，高10m）
            ax.scatter(*real_target_center, color='blue', s=300, marker='s', label='真目标', edgecolors='darkblue', linewidth=2)
            
            # 绘制真目标圆柱体
            theta = np.linspace(0, 2*np.pi, 20)
            radius = 7  # 题目指定半径7m
            height = 10  # 题目指定高10m
            
            # 圆柱体底面和顶面
            x_circle = real_target_center[0] + radius * np.cos(theta)
            y_circle = real_target_center[1] + radius * np.sin(theta)
            z_bottom = np.full_like(x_circle, real_target_center[2])
            z_top = np.full_like(x_circle, real_target_center[2] + height)
            ax.plot(x_circle, y_circle, z_bottom, 'b-', alpha=0.8, linewidth=2)
            ax.plot(x_circle, y_circle, z_top, 'b-', alpha=0.8, linewidth=2)
            
            # 圆柱体侧面线条
            for i in range(0, len(theta), 4):
                ax.plot([x_circle[i], x_circle[i]], [y_circle[i], y_circle[i]], 
                       [z_bottom[i], z_top[i]], 'b-', alpha=0.6, linewidth=1)
            
            # 更新导弹位置
            for name, data in missiles_data.items():
                if current_time <= data['flight_time']:
                    new_pos = data['pos'] + data['direction'] * missile_speed * current_time
                    missile_trajectories[name].append(new_pos.copy())
                    
                    # 绘制导弹
                    ax.scatter(*new_pos, color=data['color'], s=200, marker='^', 
                              label=f'导弹{name}', edgecolors='black', linewidth=1.5)
                    # 3D导弹编号标注
                    ax.text(new_pos[0], new_pos[1], new_pos[2]+120, name, color=data['color'], fontsize=9,
                            ha='center', va='bottom', weight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.5, edgecolor='none'))
                    
                    # 绘制导弹轨迹
                    if len(missile_trajectories[name]) > 1:
                        traj = np.array(missile_trajectories[name])
                        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                               color=data['color'], alpha=0.8, linewidth=3)
            
            # 更新无人机位置（固定航向飞行）
            for drone_name, config in drones_config.items():
                # 无人机按固定方向和速度飞行
                new_pos = config['pos'] + config['direction'] * config['speed'] * current_time
                drone_trajectories[drone_name].append(new_pos.copy())
                
                # 绘制无人机
                ax.scatter(*new_pos, color=config['color'], s=150, marker='o', 
                          label=f'无人机{drone_name}', edgecolors='black', linewidth=1.5)
                # 3D无人机编号标注
                ax.text(new_pos[0], new_pos[1], new_pos[2]+120, drone_name, color=config['color'], fontsize=9,
                        ha='center', va='bottom', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.5, edgecolor='none'))
                
                # 绘制无人机轨迹
                if len(drone_trajectories[drone_name]) > 1:
                    traj = np.array(drone_trajectories[drone_name])
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                           color=config['color'], alpha=0.8, linewidth=2, linestyle='--')
                
                # 检查是否需要投放烟幕弹
                for i, bomb in enumerate(config['bombs']):
                    bomb_id = f"{drone_name}_{i}"
                    if (current_time >= bomb['deploy_time'] and 
                        current_time < bomb['deploy_time'] + dt and
                        bomb_id not in deployed_bombs[drone_name]):
                        
                        # 在当前位置投放烟幕弹
                        bomb_pos = new_pos.copy()
                        bomb_pos[2] -= 50  # 烟幕弹投放高度偏移
                        smoke_bombs.append({
                            'pos': bomb_pos.copy(),
                            'init_pos': bomb_pos.copy(),               # 初始位置
                            'h_vel': config['direction'] * config['speed'],  # 3D下完整水平速度向量(x,y,0)
                            'deploy_time': current_time,
                            'exploded': False,
                            'explode_time': current_time + bomb['explode_delay'],
                            'drone': drone_name,
                            'bomb_id': bomb_id
                        })
                        deployed_bombs[drone_name].append(bomb_id)
                        print(f"无人机{drone_name}在时间{current_time:.1f}s投放烟幕弹{i}")
            
            # 更新烟幕弹状态
            active_smoke_count = 0
            for bomb in smoke_bombs:
                bomb_time = current_time - bomb['deploy_time']
                
                if not bomb['exploded'] and current_time >= bomb['explode_time']:
                    bomb['exploded'] = True
                    bomb['smoke_center'] = bomb['pos'].copy()  # 取爆炸瞬间(抛物线末端)位置
                    print(f"烟幕弹在时间{current_time:.1f}s爆炸")
                
                if bomb['exploded']:
                    explosion_time = current_time - bomb['explode_time']
                    if explosion_time <= smoke_effective_time:  # 有效遮蔽时间20秒
                        smoke_center = bomb['smoke_center'].copy()
                        smoke_center[2] -= smoke_drop_speed * explosion_time  # 3m/s下沉
                        active_smoke_count += 1
                        
                        # 绘制烟幕中心点
                        ax.scatter(*smoke_center, color='gray', s=300, marker='*', alpha=0.6, edgecolors='black')
                        
                        # 绘制烟幕边界球面（有效半径10m）
                        theta_smoke = np.linspace(0, 2*np.pi, 32)
                        radius_smoke = smoke_effective_radius  # 10m
                        
                        # 水平圆圈
                        x_h = smoke_center[0] + radius_smoke * np.cos(theta_smoke)
                        y_h = smoke_center[1] + radius_smoke * np.sin(theta_smoke)
                        z_h = np.full_like(x_h, smoke_center[2])
                        ax.plot(x_h, y_h, z_h, 'gray', alpha=0.7, linewidth=2)
                        
                        # 垂直圆圈
                        x_v1 = smoke_center[0] + radius_smoke * np.cos(theta_smoke)
                        y_v1 = np.full_like(x_v1, smoke_center[1])
                        z_v1 = smoke_center[2] + radius_smoke * np.sin(theta_smoke)
                        ax.plot(x_v1, y_v1, z_v1, 'gray', alpha=0.7, linewidth=2)
                else:
                    # 未爆炸：抛物线运动 (水平匀速 + 重力下降)
                    if bomb_time >= 0:
                        # 更新水平位置
                        bomb['pos'][0] = bomb['init_pos'][0] + bomb['h_vel'][0] * bomb_time
                        bomb['pos'][1] = bomb['init_pos'][1] + bomb['h_vel'][1] * bomb_time
                        # 重力作用下降 (初速度为0, z = z0 - 1/2 g t^2)
                        bomb['pos'][2] = bomb['init_pos'][2] - 0.5 * 9.8 * bomb_time**2
                        if bomb['pos'][2] < 0:
                            bomb['pos'][2] = 0  # 不低于地面
                        ax.scatter(*bomb['pos'], color='black', s=100, marker='o', edgecolors='red', linewidth=1)
            
            # 添加详细信息面板
            total_deployed = len(smoke_bombs)
            info_text = f"""3D仿真状态:{' (第二题模式)' if question2 else ''}
时间: {current_time:.1f}s
已投放烟幕弹: {total_deployed}枚
活跃烟幕团: {active_smoke_count}个
导弹飞行状态: {sum(1 for name, data in missiles_data.items() if current_time <= data['flight_time'])}/{len(missiles_data)}

题目参数:
导弹速度: 300m/s
烟幕下沉: 3m/s
有效半径: 10m
有效时间: 20s
无人机速度: 70-140m/s

操作提示: 鼠标拖拽旋转视角"""
            
            ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9),
                     verticalalignment='top', fontsize=10, fontweight='bold')
            
            # 不再自动旋转视角，保持用户设置的视角
            # ax.view_init(elev=20, azim=45 + frame * 0.5)  # 移除这行
            
            # 添加网格
            ax.grid(True, alpha=0.3)
            
            # 简化图例，避免重复
            legend_elements = []
            if current_time <= max(data['flight_time'] for data in missiles_data.values()):
                legend_elements.append(plt.Line2D([0], [0], marker='^', color='orange', label='导弹', markersize=10, linestyle='None'))
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='green', label='无人机', markersize=10, linestyle='None'))
            if active_smoke_count > 0:
                legend_elements.append(plt.Line2D([0], [0], marker='*', color='gray', label='烟幕团', markersize=15, linestyle='None'))
            if total_deployed > active_smoke_count:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='black', label='烟幕弹', markersize=8, linestyle='None'))
            
            ax.legend(handles=legend_elements, loc='upper right')
            
        except Exception as e:
            print(f"动画帧 {frame} 渲染错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("创建3D动画...")
    print("提示：您可以用鼠标拖拽来旋转3D视角！")
    
    # 创建动画
    frames = int(total_time / dt)
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=100, repeat=True, blit=False)
    
    print("3D动画创建成功！")
    return fig, ani

def create_pseudo_3d_simulation():
    """创建伪3D仿真（使用多个2D视图）"""
    print("使用伪3D模式（多视图显示）")
    
    fig = plt.figure(figsize=(20, 12))
    
    # 创建三个子图：俯视图、侧视图、正视图
    ax1 = plt.subplot(2, 2, 1)  # 俯视图 (XY)
    ax2 = plt.subplot(2, 2, 2)  # 侧视图 (XZ)  
    ax3 = plt.subplot(2, 2, 3)  # 正视图 (YZ)
    ax4 = plt.subplot(2, 2, 4)  # 信息面板
    
    # 目标点设置
    fake_target = np.array([0, 0])
    real_target = np.array([0, 200])
    
    # 导弹初始位置和数据
    missiles_data = {
        'M1': {'pos': np.array([20000, 0]), 'color': 'orange'},
        'M2': {'pos': np.array([19000, 600]), 'color': 'orangered'},
        'M3': {'pos': np.array([18000, -600]), 'color': 'red'}
    }
    
    # 无人机初始位置和飞行参数配置
    drones_config = {
        'FY1': {
            'pos': np.array([17800, 0]), 
            'color': 'green',
            'direction': np.array([-1, 0]),  # 飞行方向（单位向量）
            'speed': 120,  # m/s
            'bombs': [
                {'deploy_time': 25, 'explode_delay': 3.5},
                {'deploy_time': 30, 'explode_delay': 3.6},
                {'deploy_time': 35, 'explode_delay': 3.8}
            ]
        },
        'FY2': {
            'pos': np.array([12000, 1400]), 
            'color': 'lime',
            'direction': np.array([-0.8, -0.6]),  # 西南方向
            'speed': 120,
            'bombs': [
                {'deploy_time': 28, 'explode_delay': 3.2},
                {'deploy_time': 33, 'explode_delay': 3.7},
                {'deploy_time': 38, 'explode_delay': 3.9}
            ]
        },
        'FY3': {
            'pos': np.array([6000, -3000]), 
            'color': 'forestgreen',
            'direction': np.array([-0.6, 0.8]),  # 西北方向
            'speed': 120,
            'bombs': [
                {'deploy_time': 22, 'explode_delay': 3.3},
                {'deploy_time': 27, 'explode_delay': 3.6},
                {'deploy_time': 32, 'explode_delay': 4.0}
            ]
        },
        'FY4': {
            'pos': np.array([11000, 2000]), 
            'color': 'darkgreen',
            'direction': np.array([-0.9, -0.4]),  # 西南偏西方向
            'speed': 120,
            'bombs': [
                {'deploy_time': 26, 'explode_delay': 3.4},
                {'deploy_time': 31, 'explode_delay': 3.8},
                {'deploy_time': 36, 'explode_delay': 3.5}
            ]
        },
        'FY5': {
            'pos': np.array([13000, -2000]), 
            'color': 'lightgreen',
            'direction': np.array([-0.7, 0.7]),  # 西北方向
            'speed': 120,
            'bombs': [
                {'deploy_time': 24, 'explode_delay': 3.6},
                {'deploy_time': 29, 'explode_delay': 3.3},
                {'deploy_time': 34, 'explode_delay': 3.7}
            ]
        }
    }
    
    # 仿真参数
    missile_speed = 300  # m/s
    dt = 0.2            # 时间步长(s)
    total_time = 80     # 总仿真时间(s)
    
    # 为每个导弹计算到假目标的方向向量
    for name, data in missiles_data.items():
        direction = fake_target - data['pos']
        distance = np.linalg.norm(direction)
        data['direction'] = direction / distance
        data['flight_time'] = distance / missile_speed
    
    # 存储轨迹和烟幕弹
    missile_trajectories = {name: [data['pos'].copy()] for name, data in missiles_data.items()}
    drone_trajectories = {name: [config['pos'].copy()] for name, config in drones_config.items()}
    smoke_bombs = []  # 烟幕弹列表
    deployed_bombs = {name: [] for name in drones_config.keys()}  # 记录已投放的烟幕弹
    
    def animate(frame):
        for ax in [ax1, ax2, ax3]:
            ax.clear()
        ax4.clear()
        
        current_time = frame * 0.1
        
        # 俯视图设置
        ax1.set_xlim(-2000, 22000)
        ax1.set_ylim(-4000, 4000)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title(f'俯视图 (XY) - 时间: {current_time:.1f}s')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 侧视图设置
        ax2.set_xlim(-2000, 22000)
        ax2.set_ylim(0, 2500)
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Z (m)')
        ax2.set_title(f'侧视图 (XZ) - 时间: {current_time:.1f}s')
        ax2.grid(True, alpha=0.3)
        
        # 正视图设置
        ax3.set_xlim(-4000, 4000)
        ax3.set_ylim(0, 2500)
        ax3.set_xlabel('Y (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title(f'正视图 (YZ) - 时间: {current_time:.1f}s')
        ax3.grid(True, alpha=0.3)
        
        # 信息面板
        ax4.text(0.1, 0.9, "伪3D模式运行中", transform=ax4.transAxes, fontsize=16, fontweight='bold')
        ax4.text(0.1, 0.7, f"时间: {current_time:.1f}s", transform=ax4.transAxes, fontsize=14)
        ax4.text(0.1, 0.5, "真正的3D模式不可用", transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.3, "显示三个2D投影视图", transform=ax4.transAxes, fontsize=12)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        # 简单的点显示
        ax1.scatter(10000, 0, color='red', s=100, label='示例点')
        ax2.scatter(10000, 1000, color='red', s=100)
        ax3.scatter(0, 1000, color='red', s=100)
    
    ani = animation.FuncAnimation(fig, animate, frames=100, interval=200, repeat=True)
    return fig, ani

if __name__ == "__main__":
    print("="*60)
    print("A题：烟幕干扰弹投放策略 - 动态仿真系统")
    print("Problem A: Smoke Jamming Deployment Strategy - Dynamic Simulation")
    print("="*60)
    print("题目关键参数：")
    print("- 假目标：原点 (0,0,0)")
    print("- 真目标：圆柱形，下底面圆心(0,200,0)，半径7m，高10m")
    print("- 导弹速度：300m/s，飞向假目标")
    print("- 无人机速度：70-140m/s，等高度匀速直线飞行")
    print("- 烟幕：起爆后瞬时形成球状云团，3m/s下沉，10m半径有效遮蔽20s")
    print("- 投放间隔：每架无人机投放两枚烟幕弹至少间隔1秒")
    print("="*60)
    
    while True:
        print("\n请选择仿真模式 / Choose simulation mode:")
        print("1. 3D动态仿真 / 3D Dynamic Simulation (手动旋转)")
        print("2. 动态俯视图 / Dynamic Top View")
        print("3. 第二题模式 (仅M1与FY1) / Question 2 Mode")
        print("0. 退出 / Exit")
        
        choice = input("\n请输入选择 (0-3): ").strip()
        
        if choice == '1':
            print("\n启动3D动态仿真...")
            print("题目场景仿真：")
            print("- 3枚导弹M1、M2、M3从指定位置以300m/s飞向假目标")
            print("- 5架无人机FY1-FY5从指定位置开始飞行")
            print("- 无人机在指定时间投放烟幕弹，间隔至少1秒")
            print("- 烟幕弹重力下降，按时序起爆形成球状云团")
            print("- 烟幕云团以3m/s速度匀速下沉，20秒内有效遮蔽")
            print("- 使用鼠标拖拽旋转3D视角，滚轮缩放")
            
            try:
                fig1, ani1 = create_dynamic_simulation()
                print("显示3D仿真窗口...")
                plt.show()
            except Exception as e:
                print(f"3D仿真出错: {e}")
                import traceback
                traceback.print_exc()
            
        elif choice == '2':
            print("\n启动动态俯视图...")
            print("俯视图说明：")
            print("- 从上方观察整个战场态势")
            print("- 5架无人机各执行3次投弹任务")
            print("- 更清楚地显示平面位置关系和运动轨迹")
            print("- 烟幕显示为圆形遮蔽区域")
            try:
                fig2, ani2 = create_dynamic_top_view()
                plt.show()
            except Exception as e:
                print(f"启动2D仿真时出错: {e}")
            
        elif choice == '3':
            print("\n启动第二题模式 (仅M1与FY1)...")
            print("说明：只显示一枚导弹M1与一架无人机FY1及其投放烟幕弹效果，用于第二题分析。")
            try:
                fig3, ani3 = create_dynamic_simulation(question2=True)
                plt.show()
            except Exception as e:
                print(f"第二题3D模式出错: {e}")
                print("尝试使用俯视图...")
                try:
                    fig4, ani4 = create_dynamic_top_view(question2=True)
                    plt.show()
                except Exception as e2:
                    print(f"第二题俯视图模式出错: {e2}")
        elif choice == '0':
            print("退出程序 / Exiting program.")
            break
            
        else:
            print("无效选择，请重新输入 / Invalid choice, please try again.")