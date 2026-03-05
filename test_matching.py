"""
Test Script — Kiểm tra chính xác hệ thống nhận diện.

Các test case:
  TC1: Self-match (embedding vs chính nó) → phải đúng 100%
  TC2: Cross-student (embedding SV A vs DB) → phải match đúng SV A
  TC3: Noise simulation (mô phỏng khoảng cách xa)
       - σ=0.02 → gần (~1m), σ=0.05 → trung bình (~2-3m)
       - σ=0.10 → xa (~3-5m), σ=0.15 → rất xa (>5m)
  TC4: Margin check — thêm noise, kiểm tra có nhầm người không
  TC5: Quality-based threshold — mặt nhỏ phải bị reject nếu không rõ

Chạy: python test_matching.py
"""

import sys
import numpy as np
from collections import defaultdict

from app.database import SessionLocal
from app.models import Student, StudentPhoto
from app.face_service import (
    find_best_match_pgvector,
    find_best_match_pgvector_v2,
    _adaptive_threshold,
    _compute_face_quality,
    COSINE_THRESHOLD,
)
from app.config import (
    MATCH_MARGIN_MIN,
    TOP_K_MATCHES,
    STUDENT_AGG_TOP_N,
    QUALITY_FACE_SIZE_GOOD,
    QUALITY_FACE_SIZE_MIN,
    QUALITY_THRESHOLD_PENALTY,
)


def load_test_data(db):
    """Load all students + embeddings from DB for testing."""
    rows = (
        db.query(
            Student.id,
            Student.student_code,
            Student.full_name,
            StudentPhoto.id.label("photo_id"),
            StudentPhoto.face_embedding,
        )
        .join(StudentPhoto, Student.id == StudentPhoto.student_id)
        .all()
    )

    students = {}
    for sid, code, name, pid, emb in rows:
        if sid not in students:
            students[sid] = {
                "student_id": sid,
                "student_code": code,
                "full_name": name,
                "embeddings": [],
            }
        # pgvector returns list or numpy array
        emb_list = list(emb) if not isinstance(emb, list) else emb
        students[sid]["embeddings"].append({"photo_id": pid, "embedding": emb_list})

    return students


def add_noise(embedding, sigma):
    """Add Gaussian noise to simulate distance degradation."""
    emb = np.array(embedding, dtype=np.float64)
    noise = np.random.normal(0, sigma, size=emb.shape)
    noisy = emb + noise
    # Re-normalize to unit vector (Facenet512 embeddings are normalized)
    norm = np.linalg.norm(noisy)
    if norm > 0:
        noisy = noisy / norm
    return noisy.tolist()


def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ═══════════════════════════════════════════════════════════
#  TEST CASES
# ═══════════════════════════════════════════════════════════

def test_case_1_self_match(db, students):
    """TC1: Mỗi embedding match với chính nó → score = 1.0, đúng người."""
    print("\n" + "=" * 65)
    print("  TC1: SELF-MATCH (embedding vs chính nó)")
    print("=" * 65)

    total = 0
    correct_v1 = 0
    correct_v2 = 0

    for sid, info in students.items():
        for emb_data in info["embeddings"]:
            emb = emb_data["embedding"]
            total += 1

            # V1 (old)
            match_v1, score_v1 = find_best_match_pgvector(db, emb)
            v1_ok = match_v1 and match_v1["student_id"] == sid
            if v1_ok:
                correct_v1 += 1

            # V2 (new)
            match_v2, score_v2, debug = find_best_match_pgvector_v2(db, emb, face_quality=1.0)
            v2_ok = match_v2 and match_v2["student_id"] == sid
            if v2_ok:
                correct_v2 += 1

            status = "✓" if (v1_ok and v2_ok) else "✗"
            print(
                f"  {status} {info['full_name']} (photo {emb_data['photo_id']}): "
                f"V1={score_v1:.4f}{'✓' if v1_ok else '✗'}  "
                f"V2={score_v2:.4f}{'✓' if v2_ok else '✗'}  "
                f"margin={debug.get('margin', 0):.4f}"
            )

    print(f"\n  Kết quả TC1: V1={correct_v1}/{total} V2={correct_v2}/{total}")
    return correct_v1, correct_v2, total


def test_case_2_cross_student(db, students):
    """TC2: Mỗi embedding phải match đúng chính SV đó (không nhầm SV khác)."""
    print("\n" + "=" * 65)
    print("  TC2: CROSS-STUDENT DISCRIMINATION")
    print("=" * 65)

    total = 0
    correct_v1 = 0
    correct_v2 = 0
    wrong_v1 = []
    wrong_v2 = []

    for sid, info in students.items():
        for emb_data in info["embeddings"]:
            emb = emb_data["embedding"]
            total += 1

            # V1
            match_v1, score_v1 = find_best_match_pgvector(db, emb)
            if match_v1 and match_v1["student_id"] == sid:
                correct_v1 += 1
            elif match_v1:
                wrong_v1.append((info["full_name"], match_v1["full_name"], score_v1))

            # V2
            match_v2, score_v2, debug = find_best_match_pgvector_v2(db, emb, face_quality=1.0)
            if match_v2 and match_v2["student_id"] == sid:
                correct_v2 += 1
            elif match_v2:
                wrong_v2.append((info["full_name"], match_v2["full_name"], score_v2))

    print(f"\n  V1: {correct_v1}/{total} correct, {len(wrong_v1)} wrong")
    for actual, predicted, score in wrong_v1:
        print(f"    ✗ Thực tế: {actual} → Nhầm: {predicted} (score={score:.4f})")

    print(f"  V2: {correct_v2}/{total} correct, {len(wrong_v2)} wrong")
    for actual, predicted, score in wrong_v2:
        print(f"    ✗ Thực tế: {actual} → Nhầm: {predicted} (score={score:.4f})")

    return correct_v1, correct_v2, total


def test_case_3_noise_simulation(db, students):
    """
    TC3: Mô phỏng khoảng cách xa bằng cách thêm noise.
    Noise levels → khoảng cách ước lượng:
      σ=0.02 → ~1m (gần)
      σ=0.05 → ~2-3m (trung bình)
      σ=0.10 → ~3-5m (xa)
      σ=0.15 → ~5-7m (rất xa)
      σ=0.20 → >7m (cực xa, không nên hoạt động)
    """
    print("\n" + "=" * 65)
    print("  TC3: NOISE SIMULATION (mô phỏng khoảng cách)")
    print("=" * 65)

    noise_levels = [
        (0.02, "~1m (gần)", 1.0),
        (0.05, "~2-3m (TB)", 0.8),
        (0.10, "~3-5m (xa)", 0.5),
        (0.15, "~5-7m (rất xa)", 0.3),
        (0.20, ">7m (cực xa)", 0.1),
    ]

    np.random.seed(42)  # Reproducible
    n_trials = 5  # Mỗi embedding thử 5 lần noise

    for sigma, label, sim_quality in noise_levels:
        total = 0
        correct_v1 = 0
        correct_v2 = 0
        wrong_v1_count = 0
        wrong_v2_count = 0
        reject_v1 = 0
        reject_v2 = 0
        scores_v1 = []
        scores_v2 = []

        for sid, info in students.items():
            for emb_data in info["embeddings"]:
                emb = emb_data["embedding"]
                for trial in range(n_trials):
                    noisy_emb = add_noise(emb, sigma)
                    total += 1

                    # V1
                    match_v1, score_v1 = find_best_match_pgvector(db, noisy_emb)
                    if match_v1 and match_v1["student_id"] == sid:
                        correct_v1 += 1
                        scores_v1.append(score_v1)
                    elif match_v1:
                        wrong_v1_count += 1
                    else:
                        reject_v1 += 1

                    # V2 (with simulated quality)
                    match_v2, score_v2, debug = find_best_match_pgvector_v2(
                        db, noisy_emb, face_quality=sim_quality
                    )
                    if match_v2 and match_v2["student_id"] == sid:
                        correct_v2 += 1
                        scores_v2.append(score_v2)
                    elif match_v2:
                        wrong_v2_count += 1
                    else:
                        reject_v2 += 1

        avg_v1 = np.mean(scores_v1) if scores_v1 else 0
        avg_v2 = np.mean(scores_v2) if scores_v2 else 0

        print(f"\n  σ={sigma:.2f} ({label})  [{total} tests]")
        print(f"    V1: ✓{correct_v1} ✗{wrong_v1_count} ⊘{reject_v1}  avg_score={avg_v1:.4f}")
        print(f"    V2: ✓{correct_v2} ✗{wrong_v2_count} ⊘{reject_v2}  avg_score={avg_v2:.4f}")

        # Highlight improvement
        if wrong_v1_count > wrong_v2_count:
            diff = wrong_v1_count - wrong_v2_count
            print(f"    ★ V2 giảm {diff} lần nhầm người!")
        elif wrong_v2_count > wrong_v1_count:
            print(f"    ⚠ V2 tăng {wrong_v2_count - wrong_v1_count} lần nhầm!")


def test_case_4_margin_analysis(db, students):
    """TC4: Phân tích margin giữa SV #1 và SV #2 ở các mức noise."""
    print("\n" + "=" * 65)
    print("  TC4: MARGIN ANALYSIS (khoảng cách SV #1 vs SV #2)")
    print("=" * 65)

    noise_levels = [0.00, 0.05, 0.10, 0.15]
    np.random.seed(42)

    for sigma in noise_levels:
        margins = []
        low_margin_count = 0

        for sid, info in students.items():
            for emb_data in info["embeddings"]:
                emb = emb_data["embedding"]
                noisy_emb = add_noise(emb, sigma) if sigma > 0 else emb
                quality = max(0.1, 1.0 - sigma * 5)

                _, _, debug = find_best_match_pgvector_v2(
                    db, noisy_emb, face_quality=quality
                )
                m = debug.get("margin", 0)
                margins.append(m)
                if m < MATCH_MARGIN_MIN:
                    low_margin_count += 1

        avg_margin = np.mean(margins) if margins else 0
        min_margin = np.min(margins) if margins else 0

        print(
            f"  σ={sigma:.2f}: avg_margin={avg_margin:.4f}  "
            f"min_margin={min_margin:.4f}  "
            f"low_margin(<{MATCH_MARGIN_MIN})={low_margin_count}/{len(margins)}"
        )


def test_case_5_quality_threshold(db, students):
    """TC5: Kiểm tra adaptive threshold hoạt động đúng."""
    print("\n" + "=" * 65)
    print("  TC5: ADAPTIVE THRESHOLD")
    print("=" * 65)

    face_sizes = [30, 50, 80, 120, 200]
    for size in face_sizes:
        # Create a dummy face for quality calculation
        dummy = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        quality = _compute_face_quality(dummy, size)
        threshold = _adaptive_threshold(quality)
        print(
            f"  Face {size:3d}px → quality={quality:.3f} "
            f"→ threshold={threshold:.3f} "
            f"(base={COSINE_THRESHOLD}, penalty=+{threshold - COSINE_THRESHOLD:.3f})"
        )


def test_case_6_worst_case_attack(db, students):
    """
    TC6: Worst case — lấy embedding SV A, thêm noise hướng về SV B.
    Kiểm tra V2 có chặn được không.
    """
    print("\n" + "=" * 65)
    print("  TC6: WORST CASE — DIRECTED NOISE (nhầm có chủ đích)")
    print("=" * 65)

    student_list = list(students.values())
    if len(student_list) < 2:
        print("  ⚠ Cần ít nhất 2 sinh viên để test case này")
        return

    np.random.seed(42)
    total = 0
    wrong_v1 = 0
    wrong_v2 = 0

    for i in range(len(student_list)):
        for j in range(len(student_list)):
            if i == j:
                continue

            sv_a = student_list[i]
            sv_b = student_list[j]

            emb_a = np.array(sv_a["embeddings"][0]["embedding"])
            emb_b = np.array(sv_b["embeddings"][0]["embedding"])

            # Tạo embedding "ở giữa" A và B (mô phỏng nhầm)
            for alpha in [0.15, 0.25, 0.35]:
                # Blend: (1-alpha)*A + alpha*B → gần A nhưng lệch về B
                blended = (1 - alpha) * emb_a + alpha * emb_b
                blended = blended / (np.linalg.norm(blended) + 1e-10)
                blended = blended.tolist()
                total += 1

                # V1
                match_v1, score_v1 = find_best_match_pgvector(db, blended)
                if match_v1 and match_v1["student_id"] != sv_a["student_id"]:
                    wrong_v1 += 1

                # V2 (quality = 0.4 vì mặt xa)
                match_v2, score_v2, debug = find_best_match_pgvector_v2(
                    db, blended, face_quality=0.4
                )
                if match_v2 and match_v2["student_id"] != sv_a["student_id"]:
                    wrong_v2 += 1

                is_v1_wrong = match_v1 and match_v1["student_id"] != sv_a["student_id"]
                is_v2_wrong = match_v2 and match_v2["student_id"] != sv_a["student_id"]

                print(
                    f"  α={alpha:.2f} {sv_a['full_name']}→{sv_b['full_name']}: "
                    f"V1={'✗WRONG' if is_v1_wrong else '✓OK':>6s}(score={score_v1:.3f})  "
                    f"V2={'✗WRONG' if is_v2_wrong else '✓OK':>6s}(score={score_v2:.3f})"
                )

    print(f"\n  Kết quả TC6: V1 nhầm={wrong_v1}/{total}  V2 nhầm={wrong_v2}/{total}")
    if wrong_v1 > wrong_v2:
        print(f"  ★ V2 giảm {wrong_v1 - wrong_v2} trường hợp nhầm!")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  TEST MATCHING — KIỂM TRA ĐỘ CHÍNH XÁC                 ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    print(f"║  COSINE_THRESHOLD = {COSINE_THRESHOLD}                          ║")
    print(f"║  MATCH_MARGIN_MIN = {MATCH_MARGIN_MIN}                          ║")
    print(f"║  TOP_K_MATCHES    = {TOP_K_MATCHES}                            ║")
    print(f"║  STUDENT_AGG_TOP_N = {STUDENT_AGG_TOP_N}                            ║")
    print(f"║  QUALITY_PENALTY  = +{QUALITY_THRESHOLD_PENALTY}                         ║")
    print("╚═══════════════════════════════════════════════════════════╝")

    db = SessionLocal()
    try:
        students = load_test_data(db)
        print(f"\n  Loaded {len(students)} students from DB:")
        for sid, info in students.items():
            print(f"    [{sid}] {info['student_code']} - {info['full_name']} ({len(info['embeddings'])} photos)")

        # Run all test cases
        test_case_1_self_match(db, students)
        test_case_2_cross_student(db, students)
        test_case_3_noise_simulation(db, students)
        test_case_4_margin_analysis(db, students)
        test_case_5_quality_threshold(db, students)
        test_case_6_worst_case_attack(db, students)

        print("\n" + "=" * 65)
        print("  ĐÃ HOÀN THÀNH TẤT CẢ TEST CASES")
        print("=" * 65)

    finally:
        db.close()


if __name__ == "__main__":
    main()
