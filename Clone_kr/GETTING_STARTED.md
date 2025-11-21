# Claude Code 웹에서 LLMunix 시작하기

이 가이드는 **공개** 및 **비공개** 저장소 모두에 대해 웹의 Claude Code로 LLMunix를 설정하는 과정을 안내합니다.

## 사전 요구 사항

- GitHub 계정
- Claude Pro 또는 Max 구독 (웹의 Claude Code에 필요)
- Git 및 GitHub에 대한 기본적인 이해

## 옵션 1: 공개 저장소 (학습에 권장)

공개 저장소는 다음에 적합합니다:
- LLMunix를 배우고 실험하기
- 오픈소스 프로젝트
- 커뮤니티와 LLMunix 프로젝트 공유
- 공개적으로 다른 사람과 협업

### 단계별 설정

#### 1. 템플릿에서 저장소 생성

1. [llmunix-starter 템플릿](https://github.com/YOUR_USERNAME/llmunix-starter)으로 이동
2. 녹색 **"Use this template"** 버튼 클릭
3. **"Create a new repository"** 선택
4. 저장소 구성:
   - **Owner**: GitHub 계정 또는 조직 선택
   - **Repository name**: 이름 선택 (예: `my-llmunix-workspace`)
   - **Visibility**: **"Public"** 선택
   - **Description** (선택 사항): "동적 에이전트 기반 개발을 위한 내 LLMunix 작업 공간"
5. **"Create repository"** 클릭

#### 2. Claude Code 웹에 연결

1. [claude.ai/code](https://claude.ai/code) 방문
2. **"Connect GitHub account"** 클릭
3. Claude GitHub 앱 인증:
   - 요청된 권한 검토
   - **"All repositories"** 선택 또는 특정 저장소 선택
   - **"Install & Authorize"** 클릭

#### 3. 저장소 선택

1. Claude Code 인터페이스에서 저장소 선택기 클릭
2. 새로 생성한 저장소 찾아 선택 (예: `my-llmunix-workspace`)
3. Claude가 저장소를 안전한 클라우드 환경에 복제

#### 4. 환경 구성 (선택 사항)

기본 환경이 대부분의 경우에 작동하지만, 커스터마이즈할 수 있습니다:

1. 현재 환경 이름 클릭
2. **"Add environment"** 선택 또는 기본값 편집
3. 구성:
   - **Name**: "LLMunix Default"
   - **Network Access**: "Limited" (권장) - 패키지 관리자에 대한 접근 허용
   - **Environment Variables**: 기본적으로 필요 없음

#### 5. 첫 번째 프로젝트 시작

Claude에게 야심 차고 다면적인 목표를 제공하세요:

```
행동 분석을 사용하여 고객 이탈을 예측하는 머신러닝 파이프라인을 생성하세요.
데이터 탐색, 특성 엔지니어링, scikit-learn을 사용한 모델 훈련,
하이퍼파라미터 튜닝, 시각화가 포함된 포괄적인 평가 보고서를 포함하세요.
```

Claude가 수행할 작업:
1. `CLAUDE.md` (LLMunix 커널) 읽기
2. `projects/`에 새 프로젝트 구조 생성
3. 특화된 에이전트 동적 생성 (DataExplorationAgent, FeatureEngineerAgent 등)
4. 워크플로우 실행
5. 결과를 새 브랜치에 푸시

#### 6. 검토 및 풀 리퀘스트 생성

1. Claude가 작업을 완료하면 알림 받음
2. GitHub 인터페이스에서 변경사항 검토
3. 메인 브랜치에 병합하기 위한 풀 리퀘스트 생성
4. `projects/[ProjectName]/components/agents/`에서 동적으로 생성된 에이전트 검사
5. `projects/[ProjectName]/output/`에서 출력물 검토
6. `projects/[ProjectName]/memory/long_term/`에서 학습 내용 확인

## 옵션 2: 비공개 저장소 (프로덕션에 권장)

비공개 저장소는 다음에 이상적입니다:
- 독점 코드 및 비즈니스 로직
- 민감한 프로젝트
- 클라이언트 작업
- 프로덕션 시스템

### 단계별 설정

#### 1. 템플릿에서 비공개 저장소 생성

1. [llmunix-starter 템플릿](https://github.com/YOUR_USERNAME/llmunix-starter)으로 이동
2. 녹색 **"Use this template"** 버튼 클릭
3. **"Create a new repository"** 선택
4. 저장소 구성:
   - **Owner**: GitHub 계정 또는 조직 선택
   - **Repository name**: 이름 선택 (예: `company-llmunix-private`)
   - **Visibility**: **"Private"** 선택 ⚠️
   - **Description** (선택 사항): "비공개 LLMunix 작업 공간"
5. **"Create repository"** 클릭

#### 2. 비공개 저장소용 Claude GitHub 앱 설치

1. [claude.ai/code](https://claude.ai/code) 방문
2. 아직 연결되지 않았다면 **"Connect GitHub account"** 클릭
3. **중요**: Claude GitHub 앱에 비공개 저장소 접근 권한을 부여해야 합니다:
   - GitHub Settings → Applications → Claude (설치된 GitHub Apps 아래)로 이동
   - **"Configure"** 클릭
   - **"Repository access"** 아래에서 선택:
     - **"All repositories"** (현재 및 향후 모든 저장소에 접근 권한 부여), 또는
     - **"Only select repositories"**를 선택하고 비공개 LLMunix 저장소 추가
   - **"Save"** 클릭

#### 3. Claude Code에서 비공개 저장소 선택

1. [claude.ai/code](https://claude.ai/code)로 돌아가기
2. 저장소 선택기 클릭
3. 이제 비공개 저장소가 목록에 표시되어야 함
4. 선택 - Claude가 격리된 안전한 클라우드 VM에 복제

#### 4. 보안 고려사항을 포함한 환경 구성

비공개 저장소의 경우 더 엄격한 보안이 필요할 수 있습니다:

**옵션 A: 제한된 네트워크 접근 (기본값)**
- 패키지 관리자 및 일반적인 개발 도구 허용
- 대부분의 외부 도메인 차단
- 보안과 기능 간의 좋은 균형

**옵션 B: 네트워크 접근 없음 (최대 보안)**
1. 환경 이름 클릭 → 설정
2. **Network Access**를 **"None"**으로 설정
3. 참고: 의존성을 사전 설치하거나 SessionStart 훅을 사용해야 함

**옵션 C: 사용자 지정 허용 도메인**
1. 환경 이름 클릭 → 설정
2. **"Limited"** 접근 유지
3. Claude가 일반적인 패키지 관리자를 자동으로 허용
4. 추가 도메인의 경우, 환경별 구성 사용 고려

#### 5. 환경 변수 추가 (필요한 경우)

비공개 프로젝트의 경우 API 키나 시크릿이 필요할 수 있습니다:

1. 환경 이름 클릭 → 설정
2. **"Environment Variables"** 아래에 키-값 쌍 추가:
   ```
   DATABASE_URL=postgresql://localhost/mydb
   API_KEY=your_api_key_here
   ENVIRONMENT=development
   ```
3. **보안 참고**: 이것들은 안전하게 저장되며 격리된 VM 내에서만 접근 가능

#### 6. SessionStart 훅 구성 (선택 사항)

의존성이 있는 비공개 프로젝트의 경우, 설정 자동화:

`.claude/settings.json` 생성:
```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup",
        "hooks": [
          {
            "type": "command",
            "command": "\"$CLAUDE_PROJECT_DIR\"/scripts/setup.sh"
          }
        ]
      }
    ]
  }
}
```

`scripts/setup.sh` 생성:
```bash
#!/bin/bash

# 원격(Claude Code 웹) 환경에서만 실행
if [ "$CLAUDE_CODE_REMOTE" = "true" ]; then
  echo "원격 세션을 위한 의존성 설치 중..."

  # Python 의존성 설치
  if [ -f requirements.txt ]; then
    pip install -r requirements.txt
  fi

  # Node 의존성 설치
  if [ -f package.json ]; then
    npm install
  fi

  echo "설정 완료!"
fi

exit 0
```

실행 가능하게 만들기:
```bash
chmod +x scripts/setup.sh
```

#### 7. 비공개 프로젝트 시작

Claude에게 비즈니스 목표 제공:

```
이탈 예측 모델을 구축하기 위해 고객 데이터베이스를 분석하세요.
data/customers.csv의 데이터를 사용하세요. FastAPI를 사용하여
실시간 예측을 위한 REST API 엔드포인트를 생성하세요. 포괄적인
테스트와 API 문서를 포함하세요.
```

Claude가 비공개 코드와 함께 완전히 격리된 상태에서 작업합니다.

#### 8. 변경사항 안전하게 검토

1. 모든 작업이 격리된 Anthropic 관리 VM에서 수행됨
2. 변경사항이 비공개 저장소의 새 브랜치에 푸시됨
3. 조직 내에서 비공개로 PR 검토
4. 만족스러우면 병합

## 환경 이해

### 클라우드에서 일어나는 일

Claude Code 작업을 시작할 때:

1. **저장소 복제**: 저장소(공개 또는 비공개)가 Anthropic 관리 VM에 복제됨
2. **환경 설정**: Claude가 `CLAUDE.md`를 읽고 SessionStart 훅 실행
3. **네트워크 구성**: 인터넷 접근이 설정에 따라 구성됨
4. **격리된 실행**: Claude가 완전히 격리된 상태에서 작업
5. **안전한 푸시**: 변경사항이 안전한 GitHub 프록시를 통해 브랜치에 푸시됨

### 보안 보장

- **격리된 VM**: 각 세션이 새로운 격리된 가상 머신에서 실행
- **자격 증명 보호**: git 자격 증명이 VM에 들어가지 않음 - 인증은 안전한 프록시를 통해 범위가 지정된 자격 증명 사용
- **네트워크 제어**: Claude가 접근할 수 있는 외부 서비스를 제어
- **자동 정리**: 세션 종료 후 VM이 파괴됨

### 네트워크 접근 수준 설명

**없음**:
- 외부 네트워크 접근 없음
- 최대 보안
- 모든 의존성 사전 설치

**제한됨** (기본값):
- 패키지 관리자(npm, pip, cargo 등)에 대한 접근
- GitHub 및 일반적인 개발 도구에 대한 접근
- 대부분의 다른 외부 도메인 차단
- 대부분의 프로젝트에 권장

**전체**:
- 모든 인터넷에 대한 접근
- 필요한 경우에만 사용
- 보안 영향 검토

## 일반적인 워크플로우

### 공개 저장소 워크플로우

```
1. 템플릿 사용 → 공개 저장소 생성
2. Claude Code에 연결
3. 목표 제공 → Claude가 프로젝트 생성
4. PR 검토 → 병합
5. 커뮤니티와 학습 내용 공유
6. 추가 개발을 위해 로컬 머신에 복제
```

### 비공개 저장소 워크플로우

```
1. 템플릿 사용 → 비공개 저장소 생성
2. 비공개 접근을 위해 Claude GitHub 앱 구성
3. 시크릿을 위한 환경 변수 설정
4. 의존성을 위한 SessionStart 훅 구성
5. 목표 제공 → Claude가 격리된 상태에서 프로젝트 생성
6. 비공개로 PR 검토 → 안전하게 병합
7. 학습 내용이 비공개 저장소에 유지됨
```

### 하이브리드 워크플로우

```
1. 비공개 저장소에서 개발
2. 학습 내용 및 에이전트 템플릿 추출
3. 공유를 위한 공개 저장소 생성
4. 정제된 에이전트 및 학습 내용 게시
5. 커뮤니티가 당신의 패턴에서 혜택을 받음
```

## 웹과 로컬 간 이동

### 웹에서 로컬로 (모든 저장소 유형)

웹에서 작업을 시작한 후:

1. Claude Code 인터페이스에서 **"Open in CLI"** 버튼 클릭
2. 제공된 명령 복사
3. 로컬 터미널에서 (저장소가 체크아웃된 상태로):
   ```bash
   # Claude Code에서 명령 붙여넣기
   claude-code connect <session-id>
   ```
4. 로컬 변경사항이 스태시됨
5. 원격 세션 상태가 로드됨
6. 로컬에서 작업 계속

### 로컬에서 웹으로

변경사항을 커밋하고 푸시한 다음, 새 웹 세션을 시작하세요.

## 모범 사례

### 공개 저장소의 경우

1. **풍부하게 문서화**: `CLAUDE.md`가 공개됨 - 교육적으로 만드세요
2. **학습 내용 공유**: 다른 사람을 돕기 위해 에이전트 템플릿 커밋
3. **예제 정제**: 예제 프로젝트에서 개인 정보 제거
4. **이슈 활성화**: 커뮤니티가 버그를 보고하고 개선을 제안하도록 함
5. **라이선스 추가**: 다른 사람이 작업을 사용할 수 있는 방법 명확히

### 비공개 저장소의 경우

1. **환경 변수 사용**: 코드에 시크릿을 하드코딩하지 마세요
2. **네트워크 접근 구성**: 최소 필요 접근 수준 사용
3. **의존성 검토**: Claude가 설치하는 패키지 감사
4. **로그 감사**: 각 세션에서 Claude가 수행한 작업 검토
5. **접근 제어**: 민감한 저장소에 GitHub의 팀 권한 사용
6. **SessionStart 훅**: 안전한 환경 설정 자동화

### 둘 다의 경우

1. **명확한 목표**: 구체적이고 잘 정의된 목표 제공
2. **반복적 접근**: 단순하게 시작하고 확장
3. **에이전트 검토**: 동적으로 생성된 에이전트에서 학습
4. **메모리 통합**: 시스템이 학습하고 향상되도록 함
5. **참여 유지**: 실행 중에 Claude 모니터링 및 조정

## 문제 해결

### 문제: 비공개 저장소가 Claude Code에 나타나지 않음

**해결책**:
1. Claude GitHub 앱이 비공개 저장소에 접근할 수 있는지 확인
2. GitHub Settings → Applications → Claude로 이동
3. "Repository access" 아래에서 비공개 저장소가 선택되어 있는지 확인

### 문제: 의존성이 설치되지 않음

**해결책**:
1. SessionStart 훅 추가 (위 참조)
2. 웹 환경에서만 실행되도록 `CLAUDE_CODE_REMOTE` 확인 사용
3. 스크립트가 실행 가능한지 확인 (`chmod +x`)

### 문제: 네트워크 접근이 필요한 도메인 차단

**해결책**:
1. 도메인이 [기본 허용 목록](https://docs.claude.com/en/docs/claude-code/claude-code-on-the-web#default-allowed-domains)에 있는지 확인
2. 없으면 전체 네트워크 접근 사용 고려 (주의 필요)
3. 또는 네트워크가 제한되기 전에 SessionStart 훅을 통해 의존성 설치

### 문제: 환경 변수에 접근할 수 없음

**해결책**:
1. 환경 설정에 설정되어 있는지 확인
2. 적절한 `.env` 형식 사용: `KEY=value`
3. 코드에서 `os.getenv('KEY')` 또는 동등한 것으로 접근
4. 필요한 경우 `$CLAUDE_ENV_FILE`을 사용하여 SessionStart 훅에서 유지

### 문제: 웹에서 로컬로 세션을 이동할 수 없음

**해결책**:
- 두 곳에서 같은 GitHub 계정으로 인증되어 있는지 확인
- 저장소의 로컬 체크아웃이 있는지 확인
- git 자격 증명이 로컬에서 구성되어 있는지 확인

## 다음 단계?

설정 후:

1. **첫 번째 프로젝트 실행**: 잘 정의된 야심 찬 목표로 시작
2. **생성된 에이전트 검토**: Claude가 에이전트 프롬프트를 구성하는 방법에서 학습
3. **메모리 로그 검토**: 시스템이 무엇을 배웠는지 확인
4. **반복**: 유사한 프로젝트를 실행하고 시스템이 향상되는 것을 관찰
5. **커스터마이즈**: 필요에 따라 `CLAUDE.md` 수정
6. **공유** (공개 저장소): 커뮤니티에 학습 내용 기여

## 추가 리소스

- **메인 README**: 개요 및 철학
- **CLAUDE.md**: 커널 명세
- **MIGRATION_NOTES.md**: 최소주의 설계 이해
- **system/SmartMemory.md**: 메모리 시스템 아키텍처
- **Claude Code 웹 문서**: [공식 문서](https://docs.claude.com/en/docs/claude-code/claude-code-on-the-web)

---

**빌드할 준비가 되었습니다!** 도전적인 목표로 시작하고 LLMunix가 그것을 해결하기 위한 완벽한 에이전트 팀을 동적으로 생성하는 것을 지켜보세요.
