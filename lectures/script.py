from pytube import Playlist
from bs4 import BeautifulSoup
import requests

def get_video_title(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.title.string.split(' - YouTube')[0]
    except Exception as e:
        return f"Video {url.split('=')[-1]}"

def playlist_to_markdown(playlist_url, output_file="playlist.md"):
    p = Playlist(playlist_url)
    
    # Extract playlist title
    playlist_title = p.title
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# {playlist_title}\n\n")
        
        for i, url in enumerate(p.video_urls, 1):
            title = get_video_title(url)
            f.write(f"{i}. [{title}]({url})\n")
    
    print(f"Markdown file created: {output_file}")

# Example usage
playlist_to_markdown("https://www.youtube.com/playlist?list=PL-SaKQp0_J8jfT8r2ytzA9UlW1c4JcH4E", "michigan_deep_learning.md")

