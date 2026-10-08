"""Create the six-page Korean generalization review. Requires ReportLab.

PDF_FONT_REGULAR and PDF_FONT_BOLD may override the Windows Korean fonts.
Research proposals are distinguished from implemented or validated results.
"""
from pathlib import Path
import os
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/pdf/FU_generalization_review_2026-10-08.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)
pdfmetrics.registerFont(TTFont("KR", os.environ.get("PDF_FONT_REGULAR", "C:/Windows/Fonts/malgun.ttf")))
pdfmetrics.registerFont(TTFont("KRB", os.environ.get("PDF_FONT_BOLD", "C:/Windows/Fonts/malgunbd.ttf")))
pdfmetrics.registerFontFamily("KR", normal="KR", bold="KRB")
W, H = A4
M, CW = 42, W - 84
NAVY, TEAL = colors.HexColor("#163746"), colors.HexColor("#087F79")
INK, MUTED = colors.HexColor("#2A3E49"), colors.HexColor("#657984")
PALE, GRAY, LINE = [colors.HexColor(h) for h in ("#EAF5F2", "#F3F6F8", "#DCE5E9")]
C = canvas.Canvas(str(OUT), pagesize=A4, pageCompression=1)
C.setTitle("특정 공격을 넘어: 연합 언러닝 방어의 범용성 검토")
C.setAuthor("BadFU Defend Research")
C.setSubject("31편 문헌 지도, 근접 선행연구, 확장 가설과 검증 순서")
C.setViewerPreference("DisplayDocTitle", "true")

def para(s, x, y, w=CW, size=10, bold=False, color=INK, limit=300, draw=True):
    p = Paragraph(s, ParagraphStyle("p", fontName="KRB" if bold else "KR",
                  fontSize=size, leading=size*1.52, textColor=color, wordWrap=None))
    _, ph = p.wrap(w, H)
    if ph > limit or y+ph > H-48:
        raise ValueError(f"Paragraph overflow at {y}: {ph} {s[:70]}")
    if draw:
        p.drawOn(C, x, H-y-ph)
    return ph

def rect(x, y, w, h, fill=GRAY):
    if y+h > H-45:
        raise ValueError(f"Box overflow: {y+h}")
    C.setFillColor(fill)
    C.roundRect(x, H-y-h, w, h, 7, fill=1, stroke=0)

def rule(y):
    C.setStrokeColor(LINE)
    C.setLineWidth(.6)
    C.line(M, H-y, W-M, H-y)

def page(n, title, subtitle):
    C.setFillColor(TEAL)
    C.rect(M, H-32, 25, 3, fill=1, stroke=0)
    para("BadFU Defend / 연구 방향 재검토", M+36, 23, 320, 8.3, color=MUTED)
    para("2026.10.08", W-M-75, 23, 75, 8.3, color=MUTED)
    para(title, M, 65, size=24, bold=True, color=NAVY, limit=76)
    para(subtitle, M, 110, size=10, color=MUTED, limit=49)
    rule(H-39)
    C.setFont("KR", 8)
    C.setFillColor(MUTED)
    C.drawString(M, 21, "문헌 근거와 연구 제안 / 효과·일반화는 실험으로 확인")
    C.setFont("KRB", 8.5)
    C.setFillColor(TEAL)
    C.drawRightString(W-M, 21, f"{n} / 6")

def section(title, y):
    para(title, M, y, size=13, bold=True, color=NAVY, limit=22)
    return y+29

def callout(title, body, y, h=88):
    rect(M, y, CW, h, PALE)
    para(title, M+15, y+12, CW-30, 13.5, True, TEAL, limit=43)
    para(body, M+15, y+40, CW-30, 9.6, limit=h-49)
    return y+h+20

def table(headers, rows, widths, y, size=9.1):
    rect(M, y, CW, 29, NAVY)
    x=M
    for s,w in zip(headers,widths):
        para(s,x+9,y+7,w-18,8.7,True,colors.white,limit=16)
        x+=w
    y+=29
    for idx,row in enumerate(rows):
        heights=[para(s,0,0,w-18,size,draw=False) for s,w in zip(row,widths)]
        rh=max(heights)+19
        rect(M,y,CW,rh,GRAY if idx%2==0 else colors.white)
        x=M
        for col,(s,w) in enumerate(zip(row,widths)):
            para(s,x+9,y+9,w-18,size,col==0,limit=rh-16)
            x+=w
        y+=rh
        rule(y)
    return y

def linked(label, url):
    return f'<link href="{escape(url)}" color="#087F79">{escape(label)}</link>'

# 1. Decision, scope and present evidence.
page(1,"특정 공격을 넘어설 수 있는가", "교수님 질문에 대한 답: 현재 쌍 탐지의 한계를 인정하고, 연구 질문을 넓힌다.")
y=callout("확장 가능성은 있다. 범용성은 아직 검증되지 않았다.",
          "목표는 요청된 삭제를 이행하면서 잔존 백도어 위험을 줄이는 것.<br/>탐지만을 위한 별도 언러닝 실행은 0회로 유지한다.",160,97)
y=section("현재 방법이 특정 구조에 의존하는 이유", y)
para("반대 방향의 클라이언트 쌍이 없는 공격, 한 클라이언트 내부의 상쇄, 실제 FU 중 새로 주입되는 공격은 현재 증거를 주지 않을 수 있다. 데이터셋·seed만 늘려서는 이 한계가 사라지지 않는다.", M,y,size=10.4,limit=65)
y+=81
y=section("추천하는 확장 범위", y)
y=table(["유지할 원칙","추가로 검증할 것"],[
    ("추가 탐지 FU 0회", "기존 기록 + 실제 FU 업데이트의 변화"),
    ("요청된 삭제 이행", "위험 점수가 높아도 삭제 범위를 축소하지 않기"),
    ("선택적·제한된 정화", "같은 정상 손상 예산에서 서로 다른 공격 억제"),
    ("명확한 정보 가정", "개별 업데이트가 보이는 수평 FL 분류 모델부터")], [151,CW-151], y)
y+=22
y=section("지금 확보한 근거", y)
para("기여 차감 기반 예비 개선은 있었지만, 새 GPU 통합 실험의 실데이터 효능은 아직 없다. 테스트·CPU smoke 통과는 공격 방어 성공을 뜻하지 않는다. 새 모듈은 이 문서의 제안이며 아직 구현하지 않았다.",M,y,size=9.5,color=MUTED,limit=62)
C.showPage()

# 2. Mechanism coverage, not attack-name counting.
page(2,"공격 메커니즘으로 범위를 넓히기", "모든 공격을 하나의 ASR로 합치지 않고, 구조와 삭제 단위를 구분한다.")
y=table(["공격 / 설정","현재 쌍 신호의 한계","필요한 확장"],[
    ("FUBA / BadFU ul", "분리형 위장 삭제는<br/>첫 검증 대상", "실제 FU 효과·정상 오탐<br/>검증부터"),
    ("BadFU 본문", "같은 클라이언트 내부의<br/>상쇄가 숨을 수 있음", "샘플 단위 FU +<br/>실제 경로의 새 관측"),
    ("BadUnlearn", "FU 중 새 악성 업데이트에<br/>상쇄 상대가 필요 없음", "실제 FU 기여의<br/>시간·방향 일관성"),
    ("DBA / 모델 대체", "다자 분담 또는 직접 주입;<br/>반대 쌍이 필수가 아님", "잔존 집단 기여·다층<br/>신호의 추가 효과"),
    ("Neurotoxin", "정상 FL에 잘 덮이지 않는<br/>지속형 백도어", "장기 곡선·재상승·<br/>중간 층 검토"),
    ("Fusion backdoor", "FU 부분공간 방어와<br/>가까운 선행연구", "본문 확보 후 중복 범위<br/>확인 필수"),
    ("FedMUA", "특정 정상 샘플 오분류;<br/>트리거 ASR과 다름", "별도 타깃·그룹 손상<br/>지표로 확장 평가")], [119,194,CW-313],160,8.9)
y+=22
y=callout("BadFU의 본문과 공식 예제를 구분", "본문의 단일 클라이언트·위장 샘플 삭제와 공식 ul의 분리형 클라이언트 삭제는 다른 실험이다. 현재 GPU 패키지는 후자부터 검증한다.",y,91)
para("관측 한계: 서버에 같은 업데이트·이력·probe 출력을 제공하는 두 상황은 그 정보만으로 확실히 구별할 수 없다. secure aggregation도 현재 개별 업데이트 입력 가정 밖에 있다.", M,y,size=9.3,color=MUTED,limit=62)
C.showPage()

# 3. Proposed architecture with strict actual-path boundary.
page(3,"쌍 탐지를 하나의 증거로", "최소 확장 R+T부터 검증하고, 독립적인 효과가 있을 때만 V를 추가한다.")
y=table(["모듈","관측과 역할","한계"],[
    ("R / 요청 관계", "q와 잔존 참여자의 반대 방향·<br/>상쇄·지속성: 기존 모듈", "내부 상쇄를<br/>포괄하지 못할 수 있음"),
    ("T / 실제 FU 변화", "실제로 받은 잔존 기여를 이력 및<br/>정상 FU 변화와 비교", "정상 삭제도<br/>분포 변화를 만듦"),
    ("V / 층·출력", "실제 체크포인트의 선택 층과<br/>고정 probe 출력 변화", "clean probe에서<br/>안 보이는 공격 존재")], [105,256,CW-361],160,9.0)
y+=23
y=section("정화까지 이어지는 한 개의 실제 경로", y)
for n,title,body in [
    ("1", "요청 범위 F의 실제 FU 수행", "클라이언트 삭제와 샘플 삭제를 구분한다. 샘플 삭제 시 q 전체를 배제하지 않는다."),
    ("2", "관측 신호를 정상 요청 기준으로 결합", "쌍 점수가 낮아도 다른 증거가 작동할 수 있게 한다. 점수는 공격 확률이 아니다."),
    ("3", "잔존 의심 기여에 제한된 교정 적용", "절대·상대 노름 예산을 둔다. 정답 공격자·타깃·ASR은 운영 입력으로 쓰지 않는다."),
    ("4", "정상 손상 검사와 실제 FU 계속", "손상 기준을 넘으면 추가 교정만 취소한다. clean 손실 통과는 안전 인증이 아니다.")]:
    rect(M,y,25,25,PALE)
    para(n,M+8,y+4,18,10,True,TEAL)
    para(title,M+37,y,CW-37,10.4,True,NAVY)
    h=para(body,M+37,y+22,CW-37,9.1,limit=31)
    y+=h+39
y+=4
para("경계: 후보마다 가상 삭제·복구하는 진단 경로는 만들지 않는다. 이미 계산된 실제 FU 단계에 대한 forward 검사도 방어 비용으로 기록한다. 오프라인 비교 실험의 재학습은 운영 탐지 FU와 별개다.",M,y,size=9.1,color=MUTED,limit=60)
C.showPage()

# 4. Novelty and forgetting/security distinction.
page(4,"기존 연구보다 무엇이 더 필요한가", "무시뮬레이션·방향 검사·최근 기록·불완전 탐지 대응은 각각 이미 연구됐다.")
y=table(["가까운 선행연구","이미 있는 요소","우리의 검증 질문"],[
    ("BAMU", "출력만 보는 저비용<br/>의심 요청 탐지", "샘플 원문 없는 FL에서<br/>삭제 이행 + 정화 가능한가"),
    ("UnlearnGuard", "이력 기반 실제 FU 보호", "위장 삭제 뒤 미식별 잔존<br/>공격에 추가 이득이 있는가"),
    ("FedRecover", "저비용 복구와<br/>탐지 오류 조건 평가", "요청자와 잔존 오염 집합이<br/>다른 조건에서 효과가 있는가"),
    ("AlignIns / FLDetector", "방향·시간 일관성 검사", "정상 삭제 변화와 공격을<br/>같은 FPR에서 더 잘 나누는가"),
    ("FAUN", "최근 기록으로 방향 제거<br/>후 정상 학습", "공격자 신원을 모를 때도<br/>정화를 제어할 수 있는가"),
    ("Fusion 방어", "FU의 방향 부분공간 탐지", "본문 확인 전<br/>독창성 확정 불가")], [124,180,CW-304],160,8.7)
y+=22
y=callout("정확히 삭제해도 백도어는 남을 수 있다", "위장 데이터만 제외한 재학습에도 독성 데이터가 남을 수 있다. 망각 기준과 보안 기준을 분리하고, 정화까지 포함한 참조 학습 규칙을 정의해야 한다.",y,92)
para("망각 보장 주의: 삭제 대상의 이력으로 만든 basis·위험 점수·제어 정책은 그 정보에 의존한다. 단순 후처리라며 기존 FU 인증을 자동 상속할 수 없다. MIA·ASR·거리 중 하나만으로 완전 삭제를 주장하지 않는다.",M,y,size=9.1,color=MUTED,limit=63)
C.showPage()

# 5. Decisive experiments.
page(5,"범용성을 입증할 실험 순서", "공격 계열 하나를 개발·튜닝에서 제외하고, 고정한 방법으로 최종 시험한다.")
y=table(["순서","실험","판단"],[
    ("01 / 기반", "현재 GPU suite + 정상 대조<br/>none·zero·oracle·detected·random·retrain", "실제 FU 정화부터<br/>성립하는가"),
    ("02 / 독립 효과", "R 단독 / T 단독 / R+T / clipping<br/>같은 비용·효용 조건", "단순 조합 이상의<br/>이득이 있는가"),
    ("03 / 공격 확장", "BadUnlearn·DBA·Neurotoxin<br/>원 공격에 인위적 상쇄 쌍 추가 금지", "쌍 없는 계열로<br/>이전되는가"),
    ("04 / 삭제·FU", "단일 클라이언트 샘플 삭제<br/>독립된 두 번째 FU 백엔드", "공식 ul 또는 한<br/>백엔드에만 맞는가"),
    ("05 / 지속성", "비IID·부분 참여·연속 요청<br/>적응적 회피·장기 후속 FL", "실패 조건과<br/>재활성화는 무엇인가")], [95,291,CW-386],160,8.8)
y+=24
y=section("다섯 지표를 함께 보고한다",y)
para("<b>탐지</b> 요청 단위 FPR·TPR와 관측 부족률<br/><b>보안</b> FU 중·직후·후속 FL의 peak / mean / final ASR<br/><b>효용</b> 전체 정확도와 최악 클래스·클라이언트 손상<br/><b>망각</b> 삭제 절차·재학습 기준·MIA 진단·잔존 상태<br/><b>비용</b> 전체 시간·GPU·통신·저장·추가 forward/backward",M,y,size=9.7,limit=84)
y+=98
callout("계속할 기준도 미리 정한다", "쌍 없는 공격에서 이득이 없으면 상쇄 계열 연구로 범위를 한정한다. ASR 10–15%는 안전 보장이 아니라 시험할 목표 구간이며, 정상 FL 뒤 소멸도 별도 가설이다.",y,90)
C.showPage()

# 6. Linked reading map: all 31 papers, with short names.
page(6,"31편 문헌 지도와 읽기 우선순위", "핵심 원문은 링크로 연결했다. 세부 확인 수준과 미해결 질문은 저장소 문서에 있다.")
left=M
right=M+CW/2+12
col=CW/2-12
def group(title, entries, x, y):
    para(title,x,y,col,11.4,True,NAVY,limit=20)
    y+=28
    for label,url,note in entries:
        para(linked(label,url),x,y,col,8.9,True,limit=28)
        para(note,x,y+15,col,7.7,color=MUTED,limit=24)
        y+=35
    return y

# Space-efficient group rows keep the complete inventory readable on one page.
def compact_group(title, entries, x, y):
    para(title,x,y,col,11,True,NAVY,limit=19)
    y+=27
    for label,url in entries:
        para(linked(label,url),x,y,col,8.65,limit=27)
        y+=26
    return y

y=compact_group("A. 공격 / 10편",[
    ("BadFU (2025)","https://arxiv.org/html/2508.15541v1"),
    ("FUBA (온라인 2025 / 권호 2026)","https://ieeexplore.ieee.org/document/11231135/"),
    ("UBA-Inf (USENIX Security 2024)","https://www.usenix.org/conference/usenixsecurity24/presentation/huang-zirui"),
    ("BAMU (arXiv 2023)","https://arxiv.org/html/2310.10659v1"),
    ("BadUnlearn / UnlearnGuard (2025)","https://arxiv.org/html/2501.17396v1"),
    ("Malicious Forgetting / Fusion (2026)","https://wenwei-zhao.github.io/publication/conference-paper/3/"),
    ("FedMUA (TIFS 2025)","https://arxiv.org/html/2501.11848v1"),
    ("How To Backdoor FL (AISTATS 2020)","https://proceedings.mlr.press/v108/bagdasaryan20a.html"),
    ("DBA (ICLR 2020)","https://openreview.net/forum?id=rkgyS0VFvr"),
    ("Neurotoxin (ICML 2022)","https://proceedings.mlr.press/v162/zhang22w.html")],left,156)
compact_group("B. 탐지·정화 / 9편",[
    ("MASA (WACV 2025)","https://arxiv.org/html/2411.01040v1"),
    ("FoolsGold (RAID 2020)","https://www.usenix.org/conference/raid2020/presentation/fung"),
    ("FLDetector (KDD 2022)","https://arxiv.org/abs/2207.09209"),
    ("FLAME (USENIX Security 2022)","https://www.usenix.org/conference/usenixsecurity22/presentation/nguyen"),
    ("FLTrust (NDSS 2021)","https://arxiv.org/abs/2012.13995"),
    ("DeepSight (NDSS 2022)","https://arxiv.org/abs/2201.00763"),
    ("RNP (ICML 2023)","https://proceedings.mlr.press/v202/li23v.html"),
    ("SCRUB-FL (arXiv 2026)","https://arxiv.org/html/2606.22700v1"),
    ("AlignIns (CVPR 2025)","https://arxiv.org/html/2503.07978v1")],left,y+14)
y=compact_group("C. FU·복구 / 7편",[
    ("FedEraser (IWQOS 2021)","https://www.cs.sfu.ca/~jcliu/Papers/FedEraser21.pdf"),
    ("FedRecover (IEEE S&P 2023)","https://arxiv.org/html/2210.10936v1"),
    ("Crab (arXiv 2024)","https://arxiv.org/abs/2401.08216"),
    ("SIFU (AISTATS 2024)","https://proceedings.mlr.press/v238/fraboni24a.html"),
    ("Machine Unlearning / SISA (2021)","https://arxiv.org/abs/1912.03817"),
    ("GDFA (CVPR 2026)","https://openaccess.thecvf.com/content/CVPR2026/html/Weng_GDFA_Geometry-Driven_Federated_Unlearning_with_Directional_Task_Vector_Alignment_CVPR_2026_paper.html"),
    ("FAUN (ICASSP 2026)","https://arxiv.org/html/2605.02110v1")],right,156)
y=compact_group("D. 망각·설계 / 5편",[
    ("Certified Data Removal (ICML 2020)","https://proceedings.mlr.press/v119/guo20c.html"),
    ("Auditable Definitions (Security 2022)","https://www.usenix.org/conference/usenixsecurity22/presentation/thudi"),
    ("Gradient Projection Memory (2021)","https://arxiv.org/abs/2103.09762"),
    ("On Calibration (ICML 2017)","https://proceedings.mlr.press/v70/guo17a.html"),
    ("Gone but Not Forgotten (SEI 2024)","https://www.sei.cmu.edu/library/gone-not-forgotten-improved-benchmarks-machine-unlearning/")],right,y+14)
rect(right,y+15,col,151,PALE)
para("먼저 정밀 비교할 5개",right+12,y+27,col-24,10.5,True,TEAL)
para("UnlearnGuard · FAUN · AlignIns<br/>FedRecover · Fusion 방어",right+12,y+52,col-24,9.1,limit=30)
para("미해결: Fusion·FUBA 출판 본문,<br/>GDFA 본문 접근과 공식 코드 대조.<br/>31편 모두의 전체 본문·코드를<br/>검증했다는 뜻은 아니다.",right+12,y+89,col-24,8.5,color=MUTED,limit=55)
para("추천 순서: 공격·삭제 단위 → 가까운 방어 → 비상쇄 공격 → 망각 정의와 평가",M,754,size=8.9,color=MUTED,limit=20)
C.showPage()
C.save()
print(OUT)
