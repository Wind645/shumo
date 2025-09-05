import matplotlib.pyplot as plt
from viz.top_view import create_dynamic_top_view
from viz.viz3d import create_dynamic_simulation


def main():
    try:
        mode = input("选择模式: 1=3D, 2=俯视图 (默认2): ").strip() or '2'
    except Exception:
        mode = '2'
    fig, _ = (create_dynamic_simulation() if mode == '1' else create_dynamic_top_view())
    plt.show()


if __name__ == "__main__":
    main()
