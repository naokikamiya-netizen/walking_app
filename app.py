# ==========================================================
#  app.py - 歩行分析アプリのメインプログラム
# ==========================================================
import streamlit as st
from scipy.signal import find_peaks
import cv2
import mediapipe as mp
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import japanize_matplotlib

# --- メインの分析ロジックを関数化 ---
def analyze_walking(video_path, progress_bar, status_text):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("エラー: 動画ファイルを開けませんでした。")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    all_angles, all_images = [], []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_count in range(total_frames):
        success, image = cap.read()
        if not success: break
        
        progress_percentage = (frame_count + 1) / total_frames
        progress_bar.progress(progress_percentage)
        status_text.text(f"フレーム分析中: {frame_count + 1}/{total_frames}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
        
        current_angle = all_angles[-1] if all_angles else 0
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            try:
                p_ls, p_rs = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                p_lh, p_rh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                shoulder_center_x, shoulder_center_y = (p_ls.x + p_rs.x) / 2, (p_ls.y + p_rs.y) / 2
                hip_center_x, hip_center_y = (p_lh.x + p_rh.x) / 2, (p_lh.y + p_rh.y) / 2
                v_x, v_y = shoulder_center_x - hip_center_x, hip_center_y - shoulder_center_y
                angle = -math.degrees(math.atan2(v_x, v_y))
                if abs(angle) < 45: current_angle = angle
            except Exception: pass
        
        all_angles.append(current_angle)
        all_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Matplotlib用にRGBで保存
    
    status_text.text("分析完了。サマリーを作成中...")
    cap.release()

    summary = {}
    if len(all_angles) > int(fps):
        angles_np = np.array(all_angles)
        CORE_SWING_PERCENTAGE_THRESHOLD = 0.7 
        peak_distance = int(fps * 0.4); peak_height = 2.0
        left_peak_indices, _ = find_peaks(angles_np, height=peak_height, distance=peak_distance)
        right_peak_indices, _ = find_peaks(-angles_np, height=peak_height, distance=peak_distance)
        all_core_swing_angles_left, all_core_swing_angles_right = [], []

        for p_idx in left_peak_indices:
            peak_value = angles_np[p_idx]; threshold = peak_value * CORE_SWING_PERCENTAGE_THRESHOLD
            start_idx, end_idx = p_idx, p_idx
            while start_idx > 0 and angles_np[start_idx-1] > threshold: start_idx -= 1
            while end_idx < len(angles_np)-1 and angles_np[end_idx+1] > threshold: end_idx += 1
            all_core_swing_angles_left.extend(angles_np[start_idx:end_idx+1])

        for p_idx in right_peak_indices:
            peak_value = angles_np[p_idx]; threshold = peak_value * CORE_SWING_PERCENTAGE_THRESHOLD
            start_idx, end_idx = p_idx, p_idx
            while start_idx > 0 and angles_np[start_idx-1] < threshold: start_idx -= 1
            while end_idx < len(angles_np)-1 and angles_np[end_idx+1] < threshold: end_idx += 1
            all_core_swing_angles_right.extend(angles_np[start_idx:end_idx+1])

        summary = {
            'avg_swing_core_left': np.mean(all_core_swing_angles_left) if all_core_swing_angles_left else 0,
            'avg_swing_core_right': np.mean(all_core_swing_angles_right) if all_core_swing_angles_right else 0,
        }
    else:
        return None, None

    status_text.text("結果の動画を生成中...")
    frame_h, frame_w, _ = all_images[0].shape
    graph_h, summary_h = 350, 300
    final_h, final_w = frame_h + graph_h + summary_h, frame_w
    
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_video_path = temp_output.name
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (final_w, final_h))
    
    fig_s, ax_s = plt.subplots(figsize=(final_w/100, summary_h/100), dpi=100, facecolor='#1E1E1E')
    ax_s.axis('off'); ax_s.set_title('歩行分析サマリー', color='white', fontsize=28, pad=25, weight='bold')
    texts_left = [(0.1, 0.65, "平均左スイング側屈:", '#33FF57', 28), (0.1, 0.35, "平均右スイング側屈:", '#33A8FF', 28)]
    texts_right = [(0.9, 0.65, f"{summary['avg_swing_core_left']:.2f} 度", '#33FF57', 28), (0.9, 0.35, f"{abs(summary['avg_swing_core_right']):.2f} 度", '#33A8FF', 28)]
    for x, y, text, color, size in texts_left: ax_s.text(x, y, text, color=color, fontsize=size, ha='left', va='center', transform=ax_s.transAxes, weight='bold')
    for x, y, text, color, size in texts_right: ax_s.text(x, y, text, color=color, fontsize=size, ha='right', va='center', transform=ax_s.transAxes, weight='bold')
    fig_s.canvas.draw(); summary_img = cv2.cvtColor(np.asarray(fig_s.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR); plt.close(fig_s)
    
    time_stamps = np.arange(len(all_angles)) / fps
    for i, rgb_image in enumerate(all_images):
        progress_percentage = (i + 1) / len(all_images)
        if i % 10 == 0: # 10フレームごとに進捗を更新
            progress_bar.progress(progress_percentage)

        fig_g, ax_g = plt.subplots(figsize=(final_w/100, graph_h/100), dpi=100)
        fig_g.set_facecolor('#1E1E1E'); ax_g.set_facecolor('#1E1E1E')
        ax_g.tick_params(colors='white', labelsize=14)
        for spine in ax_g.spines.values(): spine.set_edgecolor('white')
        ax_g.plot(time_stamps, all_angles, color='#00FFFF', lw=2.5)
        ax_g.plot(time_stamps[i], all_angles[i], 'o', markersize=12, color='#FF1493', mec='white')
        ax_g.axhline(0, color='red', linestyle='--', lw=2)
        ax_g.set_title('体幹の側屈（リアルタイム）', color='white', fontsize=24, pad=20)
        ax_g.set_xlabel('時間 (秒)', color='white', fontsize=18); ax_g.set_ylabel('側屈角度 (度)', color='white', fontsize=18)
        y_abs_max = max(abs(val) for val in all_angles) if all_angles else 10
        y_limit = max(y_abs_max, 10) * 1.2
        ax_g.set_ylim(-y_limit, y_limit)
        ax_g.text(1.02, 0.9, '▲ 左側屈', transform=ax_g.transAxes, color='#33FF57', va='center', fontsize=16, weight='bold')
        ax_g.text(1.02, 0.1, '▼ 右側屈', transform=ax_g.transAxes, color='#33A8FF', va='center', fontsize=16, weight='bold')
        ax_g.grid(True, linestyle=':', color='gray', alpha=0.7); fig_g.tight_layout(pad=2.5)
        fig_g.canvas.draw(); graph_img = cv2.cvtColor(np.asarray(fig_g.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR); plt.close(fig_g)
        out.write(cv2.vconcat([cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR), graph_img, summary_img])) # BGRに戻して書き込み
    
    out.release()
    status_text.text("完了！")
    return output_video_path, summary

# --- StreamlitのUI部分 ---
st.set_page_config(page_title="歩行分析アプリ", layout="wide")
st.title("🚶‍♂️ 歩行分析アプリ")
st.write("---")
st.write("スマートフォンで撮影した歩行動画をアップロードするだけで、体幹の側屈を自動で分析し、グラフ付きの動画を生成します。")

uploaded_file = st.file_uploader("ここに動画ファイル（mp4, movなど）をドラッグ＆ドロップしてください", type=["mp4", "mov", "avi", "m4v"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name

    st.video(temp_video_path)

    if st.button("この動画を分析する", type="primary"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        with st.spinner('分析を実行中です...しばらくお待ちください。'):
            output_video_path, summary = analyze_walking(temp_video_path, progress_bar, status_text)

        if output_video_path and summary:
            st.success("🎉 分析が完了しました！")
            st.balloons()
            
            st.subheader("分析結果サマリー")
            col1, col2 = st.columns(2)
            col1.metric(label="平均左スイング側屈", value=f"{summary['avg_swing_core_left']:.2f} 度")
            col2.metric(label="平均右スイング側屈", value=f"{abs(summary['avg_swing_core_right']):.2f} 度")

            st.subheader("分析結果ビデオ")
            # Streamlitで正しく表示するために動画を再読み込み
            video_file = open(output_video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

            st.download_button(
                label="結果のビデオをダウンロード",
                data=video_bytes,
                file_name="result.mp4",
                mime="video/mp4"
            )
            
            os.remove(output_video_path)
        else:
            st.error("分析中にエラーが発生しました。別の動画で試すか、動画が短すぎないか確認してください。")
        
        os.remove(temp_video_path)