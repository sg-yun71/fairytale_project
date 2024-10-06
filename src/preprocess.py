import os
import json

def load_raw_data(data_dir):
    """
    데이터 폴더 내의 모든 JSON 파일을 읽어오는 함수
    :param data_dir: 데이터가 저장된 폴더 경로
    :return: 통합된 동화 데이터 리스트
    """
    stories = []

    # 폴더 내 모든 JSON 파일을 순회하며 데이터 추출
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                # 파일 경로 출력 (디버깅용)
                print(f"읽는 중: {file_path}")

                # JSON 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        
                        # 'title'과 'paragraphInfo'가 있는지 확인하고 추출
                        title = data.get('title', '제목 없음')
                        paragraph_info = data.get('paragraphInfo', [])

                        # 'srcText' 추출하여 내용을 하나로 합침
                        content = ' '.join([paragraph.get('srcText', '') for paragraph in paragraph_info])

                        # 추출한 데이터를 리스트에 추가
                        stories.append({
                            'title': title.strip(),
                            'description': content.strip()
                        })

                    except json.JSONDecodeError as e:
                        print(f"JSON 디코딩 오류: {e}, 파일: {file_path}")

    return stories


def save_processed_data(stories, output_file):
    """
    전처리된 데이터를 JSON 파일로 저장하는 함수
    :param stories: 전처리된 동화 데이터 리스트
    :param output_file: 저장할 파일 경로
    """
    # 추출한 데이터를 JSON 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stories, f, ensure_ascii=False, indent=4)

    print(f"{output_file}에 {len(stories)}개의 동화 데이터가 저장되었습니다.")


def run_preprocessing():
    """
    데이터 전처리 실행 함수
    """
    # 데이터 파일 경로 설정
    data_dir = './data/'  # 모든 JSON 파일이 포함된 최상위 폴더 경로
    output_file = './data/processed_stories.json'  # 전처리된 데이터를 저장할 파일

    # 데이터 로드 및 전처리
    stories = load_raw_data(data_dir)

    # 전처리 결과 저장
    save_processed_data(stories, output_file)


if __name__ == "__main__":
    run_preprocessing()