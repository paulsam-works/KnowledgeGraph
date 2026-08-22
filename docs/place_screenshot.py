"""Copy the portal screenshot into docs/images/search_portal.jpg. Run: python docs/place_screenshot.py"""
from pathlib import Path
import shutil

src = Path(
    r"C:\Users\Samrat\.cursor\projects\c-Users-Samrat-Sams-Den-Sam-Workspace-GenAI-Workspace-projects-KnowledgeGraph\assets\c__Users_Samrat_AppData_Roaming_Cursor_User_workspaceStorage_35b88d444f444c7472a71ab9c78bd45d_images_search_portal-e2768ce9-c6c6-4aa2-a6b0-528e9f04abc4.jpg"
)
dst = Path(__file__).resolve().parent / "images" / "search_portal.jpg"
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, dst)
print("Wrote", dst)
