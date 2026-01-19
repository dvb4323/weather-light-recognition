import pandas as pd
import matplotlib.pyplot as plt
import argparse

# --- CẤU HÌNH STYLE (BOX STYLE) ---
def set_box_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.linewidth'] = 1.0 # Độ dày khung
    
    # [SỬA] BẬT KHUNG TRÊN VÀ PHẢI ĐỂ ĐÓNG HỘP
    plt.rcParams['axes.spines.top'] = True
    plt.rcParams['axes.spines.right'] = True
    
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = ':'

def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"✅ Đã lưu ảnh: {filename}")
    plt.close(fig)

def plot_training_history(log_file='training_log.csv'):
    # 1. Đọc dữ liệu
    try:
        df = pd.read_csv(log_file)
        print("Đọc thành công file log. Các cột:", df.columns.tolist())
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file {log_file}!")
        return

    set_box_style()

    # --- CẤU HÌNH TRỤC EPOCH ---
    epoch_ticks = list(range(0, 21, 4))  
    epoch_limit = (0, 21) 

    # --- BIỂU ĐỒ 1: LOSS ---
    fig1, ax1 = plt.subplots(figsize=(5, 4)) 
    
    ax1.plot(df['Epoch'], df['Train_Loss'], label='Training', color='#D55E00', marker='o', markersize=4) 
    ax1.plot(df['Epoch'], df['Val_Loss'], label='Validation', color='#0072B2', marker='o', markersize=4) 
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend(frameon=True) # [Tùy chọn] Bật khung cho chú thích nếu thích
    
    ax1.set_xticks(epoch_ticks)
    ax1.set_xlim(epoch_limit)
    ax1.set_ylim(bottom=0)
    
    save_plot(fig1, 'plot_loss.png')

    # --- XÁC ĐỊNH TRỤC ACCURACY CHUNG ---
    all_acc = pd.concat([
        df['Train_Weather_Acc'], df['Val_Weather_Acc'],
        df['Train_Time_Acc'], df['Val_Time_Acc']
    ])
    
    is_percent = all_acc.max() > 1.0
    y_max = 105 if is_percent else 1.05
    y_min = 0 
    
    # --- BIỂU ĐỒ 2: WEATHER ACCURACY ---
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    
    ax2.plot(df['Epoch'], df['Train_Weather_Acc'], label='Training', color='#009E73', marker='o', markersize=4) 
    ax2.plot(df['Epoch'], df['Val_Weather_Acc'], label='Validation', color='#CC79A7', marker='o', markersize=4) 
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)' if is_percent else 'Accuracy')
    ax2.set_title('Weather Classification Accuracy')
    ax2.legend(frameon=True)
    
    ax2.set_xticks(epoch_ticks)
    ax2.set_xlim(epoch_limit)
    ax2.set_ylim(y_min, y_max)
    
    save_plot(fig2, 'plot_weather_acc.png')

    # --- BIỂU ĐỒ 3: TIME ACCURACY ---
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    
    ax3.plot(df['Epoch'], df['Train_Time_Acc'], label='Training', color='#E69F00', marker='o', markersize=4) 
    ax3.plot(df['Epoch'], df['Val_Time_Acc'], label='Validation', color='#56B4E9', marker='o', markersize=4) 
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)' if is_percent else 'Accuracy')
    ax3.set_title('Time Classification Accuracy')
    ax3.legend(frameon=True)
    
    ax3.set_xticks(epoch_ticks)
    ax3.set_xlim(epoch_limit)
    ax3.set_ylim(y_min, y_max)
    
    save_plot(fig3, 'plot_time_acc.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, default='training_log.csv', help='Path to log file')
    args = parser.parse_args()
    
    plot_training_history(args.log)