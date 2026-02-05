#!/usr/bin/env python3
"""
Database management CLI for Creator Catalyst.
Provides command-line tools for managing the database.
"""

import argparse
import sys
import os
from datetime import datetime
from pathlib import Path

# --- FIX: Add project root to sys.path ---
# This ensures Python can find the 'src' folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import application modules
from src.database.credits_manager import get_credits_manager
from src.database.database import get_database, Database
from src.database.storage_manager import get_storage_manager


def cmd_init(args):
    """Initialize database."""
    db = get_database(args.database)
    print(f"✅ Database initialized at: {args.database}")


def cmd_stats(args):
    """Show database statistics."""
    storage = get_storage_manager(args.database)
    stats = storage.get_statistics()
    
    print("=" * 60)
    print("DATABASE STATISTICS")
    print("=" * 60)
    print(f"Total Videos:     {stats['total_videos']}")
    print(f"Total Content:    {stats['total_contents']}")
    print()
    print("Content by Type:")
    for content_type, count in stats['content_by_type'].items():
        print(f"  {content_type:20s}: {count}")
    print()
    print("Average Grounding Rates:")
    rates = stats['average_grounding_rates']
    print(f"  Blog:             {rates['blog']:.1%}")
    print(f"  Social:           {rates['social']:.1%}")
    print(f"  Shorts:           {rates['shorts']:.1%}")
    print("=" * 60)


def cmd_list(args):
    """List all videos."""
    storage = get_storage_manager(args.database)
    videos = storage.get_all_videos_summary(limit=args.limit)
    
    if not videos:
        print("No videos found.")
        return
    
    print("=" * 100)
    print(f"{'ID':<5} {'Filename':<40} {'Platform':<12} {'Status':<12} {'Uploaded':<20}")
    print("=" * 100)
    
    for video in videos:
        uploaded = datetime.fromisoformat(video['uploaded_at'])
        print(f"{video['id']:<5} {video['filename'][:38]:<40} {video['platform']:<12} "
              f"{video['processing_status']:<12} {uploaded.strftime('%Y-%m-%d %H:%M'):<20}")
    
    print("=" * 100)
    print(f"Total: {len(videos)} videos")


def cmd_show(args):
    """Show details for a specific video."""
    storage = get_storage_manager(args.database)
    results = storage.load_video_results(args.video_id)
    
    if not results:
        print(f"❌ Video {args.video_id} not found")
        return
    
    video = results['video']
    
    print("=" * 60)
    print(f"VIDEO DETAILS - ID: {video['id']}")
    print("=" * 60)
    print(f"Filename:         {video['filename']}")
    print(f"Platform:         {video['platform']}")
    print(f"File Size:        {video['file_size_mb']:.2f} MB")
    print(f"Uploaded:         {video['uploaded_at']}")
    print(f"Status:           {video['processing_status']}")
    print(f"Grounding:        {'Enabled' if video['grounding_enabled'] else 'Disabled'}")
    print()
    
    print("Content Generated:")
    if results.get('captions'):
        print(f"  ✅ Captions ({len(results['captions'])} chars)")
    if results.get('blog_post'):
        print(f"  ✅ Blog Post ({len(results['blog_post'])} chars)")
    if results.get('social_post'):
        print(f"  ✅ Social Post ({len(results['social_post'])} chars)")
    if results.get('shorts_ideas'):
        print(f"  ✅ Shorts Ideas ({len(results['shorts_ideas'])} ideas)")
    if results.get('thumbnail_ideas'):
        print(f"  ✅ Thumbnails ({len(results['thumbnail_ideas'])} ideas)")
    
    if results.get('grounding_report'):
        report = results['grounding_report']
        print()
        print("Grounding Report:")
        print(f"  Blog Rate:        {report['blog_grounding_rate']:.1%}")
        print(f"  Social Rate:      {report['social_grounding_rate']:.1%}")
        print(f"  Shorts Rate:      {report['shorts_verification_rate']:.1%}")
        print(f"  Verified Claims:  {report['verified_claims']}/{report['total_claims']}")
    
    print("=" * 60)


def cmd_export(args):
    """Export video results to JSON."""
    storage = get_storage_manager(args.database)
    
    output_path = args.output or f"export_video_{args.video_id}.json"
    storage.export_video_results(args.video_id, output_path)
    print(f"✅ Exported to: {output_path}")


def cmd_import(args):
    """Import video results from JSON."""
    storage = get_storage_manager(args.database)
    
    video_id = storage.import_video_results(args.json_file)
    print(f"✅ Imported with video ID: {video_id}")


def cmd_delete(args):
    """Delete a video and all its content."""
    storage = get_storage_manager(args.database)
    
    if not args.force:
        response = input(f"⚠️  Delete video {args.video_id} and ALL content? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return
    
    storage.delete_video_and_content(args.video_id)
    print(f"✅ Video {args.video_id} deleted")


def cmd_search(args):
    """Search videos by filename OR semantic content."""
    storage = get_storage_manager(args.database)
    
    # This queries both 'filename' and the new 'searchable_text' metadata blob
    videos = storage.db.search_videos(args.query, limit=50)
    
    if not videos:
        print(f"No videos found matching '{args.query}'")
        return
    
    print("=" * 100)
    print(f"SEMANTIC SEARCH RESULTS FOR: '{args.query}'")
    print("=" * 100)
    print(f"{'ID':<5} {'Filename':<40} {'Platform':<12} {'Status':<12} {'Uploaded':<20}")
    print("=" * 100)
    
    for video in videos:
        uploaded = datetime.fromisoformat(video.uploaded_at)
        print(f"{video.id:<5} {video.filename[:38]:<40} {video.platform:<12} "
              f"{video.processing_status:<12} {uploaded.strftime('%Y-%m-%d %H:%M'):<20}")
    
    print("=" * 100)
    print(f"Found: {len(videos)} videos (Matched via Filename or Generated Content)")


def cmd_recent(args):
    """Show recent activity."""
    storage = get_storage_manager(args.database)
    activities = storage.get_recent_activity(limit=args.limit)
    
    if not activities:
        print("No recent activity.")
        return
    
    print("=" * 100)
    print("RECENT ACTIVITY")
    print("=" * 100)
    print(f"{'Video ID':<10} {'Content Type':<20} {'Filename':<40} {'Created':<20}")
    print("=" * 100)
    
    for activity in activities:
        created = datetime.fromisoformat(activity['created_at'])
        print(f"{activity['video_id']:<10} {activity['content_type']:<20} "
              f"{activity['filename'][:38]:<40} {created.strftime('%Y-%m-%d %H:%M'):<20}")
    
    print("=" * 100)


def cmd_cleanup(args):
    """Clean up orphaned records."""
    db = get_database(args.database)
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        # Find videos with no content
        cursor.execute("""
            SELECT v.id, v.filename 
            FROM videos v
            LEFT JOIN content_outputs c ON v.id = c.video_id
            WHERE c.id IS NULL
        """)
        
        orphaned = cursor.fetchall()
        
        if not orphaned:
            print("✅ No orphaned records found")
            return
        
        print(f"Found {len(orphaned)} videos with no content:")
        for row in orphaned:
            print(f"  Video {row['id']}: {row['filename']}")
        
        if not args.force:
            response = input("Delete these videos? (yes/no): ")
            if response.lower() != 'yes':
                print("Cancelled.")
                return
        
        for row in orphaned:
            cursor.execute("DELETE FROM videos WHERE id = ?", (row['id'],))
        
        print(f"✅ Cleaned up {len(orphaned)} orphaned videos")


def cmd_credits_balance(args):
    """Show credit balance."""
    credits = get_credits_manager(args.database)
    stats = credits.get_user_stats(args.user_id)
    
    print("=" * 60)
    print("CREDIT BALANCE")
    print("=" * 60)
    print(f"Current Balance:  {stats['current_balance']} credits")
    print(f"Total Earned:     {stats['total_earned']} credits")
    print(f"Total Spent:      {stats['total_spent']} credits")
    print(f"Last Updated:     {stats['last_updated']}")
    print("=" * 60)


def cmd_credits_add(args):
    """Add credits to account."""
    credits = get_credits_manager(args.database)
    new_balance = credits.add_credits(
        args.amount,
        user_id=args.user_id,
        description=args.description or f"Added {args.amount} credits"
    )
    print(f"✅ Added {args.amount} credits. New balance: {new_balance}")


def cmd_credits_history(args):
    """Show credit transaction history."""
    credits = get_credits_manager(args.database)
    transactions = credits.get_transaction_history(user_id=args.user_id, limit=args.limit)
    
    if not transactions:
        print("No transactions found.")
        return
    
    print("=" * 100)
    print("CREDIT TRANSACTION HISTORY")
    print("=" * 100)
    print(f"{'Date':<20} {'Type':<8} {'Amount':<8} {'Balance':<10} {'Description':<50}")
    print("=" * 100)
    
    for txn in transactions:
        created = datetime.fromisoformat(txn['created_at'])
        txn_type = "+" if txn['type'] == 'credit' else "-"
        print(f"{created.strftime('%Y-%m-%d %H:%M:%S'):<20} "
              f"{txn_type}{txn['amount']:<7} "
              f"{txn['balance_after']:<10} "
              f"{txn['description'][:48]:<50}")
    
    print("=" * 100)


def cmd_credits_reset(args):
    """Reset credits to default amount."""
    credits = get_credits_manager(args.database)
    
    if not args.force:
        response = input(f"⚠️  Reset credits to {args.amount or 'default'}? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return
    
    credits.reset_credits(user_id=args.user_id, new_balance=args.amount)
    print(f"✅ Credits reset for user: {args.user_id}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Creator Catalyst Database Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s stats                    # Show database statistics
  %(prog)s list                     # List all videos
  %(prog)s show 123                 # Show details for video 123
  %(prog)s export 123 -o video.json # Export video 123
  %(prog)s import video.json        # Import from JSON
  %(prog)s search "my video"        # Search for videos
  %(prog)s delete 123               # Delete video 123
        """
    )
    
    parser.add_argument(
        '-d', '--database',
        default='creator_catalyst.db',
        help='Path to database file (default: creator_catalyst.db)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # init
    parser_init = subparsers.add_parser('init', help='Initialize database')
    parser_init.set_defaults(func=cmd_init)
    
    # stats
    parser_stats = subparsers.add_parser('stats', help='Show statistics')
    parser_stats.set_defaults(func=cmd_stats)
    
    # list
    parser_list = subparsers.add_parser('list', help='List all videos')
    parser_list.add_argument('-l', '--limit', type=int, default=100, help='Limit results')
    parser_list.set_defaults(func=cmd_list)
    
    # show
    parser_show = subparsers.add_parser('show', help='Show video details')
    parser_show.add_argument('video_id', type=int, help='Video ID')
    parser_show.set_defaults(func=cmd_show)
    
    # export
    parser_export = subparsers.add_parser('export', help='Export video to JSON')
    parser_export.add_argument('video_id', type=int, help='Video ID')
    parser_export.add_argument('-o', '--output', help='Output file path')
    parser_export.set_defaults(func=cmd_export)
    
    # import
    parser_import = subparsers.add_parser('import', help='Import video from JSON')
    parser_import.add_argument('json_file', help='JSON file path')
    parser_import.set_defaults(func=cmd_import)
    
    # delete
    parser_delete = subparsers.add_parser('delete', help='Delete video')
    parser_delete.add_argument('video_id', type=int, help='Video ID')
    parser_delete.add_argument('-f', '--force', action='store_true', help='Force without confirmation')
    parser_delete.set_defaults(func=cmd_delete)
    
    # search
    parser_search = subparsers.add_parser('search', help='Search videos')
    parser_search.add_argument('query', help='Search query')
    parser_search.set_defaults(func=cmd_search)
    
    # recent
    parser_recent = subparsers.add_parser('recent', help='Show recent activity')
    parser_recent.add_argument('-l', '--limit', type=int, default=20, help='Limit results')
    parser_recent.set_defaults(func=cmd_recent)
    
    # cleanup
    parser_cleanup = subparsers.add_parser('cleanup', help='Clean up orphaned records')
    parser_cleanup.add_argument('-f', '--force', action='store_true', help='Force without confirmation')
    parser_cleanup.set_defaults(func=cmd_cleanup)

    # NEW: credits-balance
    parser_credits_bal = subparsers.add_parser('credits-balance', help='Show credit balance')
    parser_credits_bal.add_argument('-u', '--user-id', default='default_user', help='User ID')
    parser_credits_bal.set_defaults(func=cmd_credits_balance)

    # NEW: credits-add
    parser_credits_add = subparsers.add_parser('credits-add', help='Add credits')
    parser_credits_add.add_argument('amount', type=int, help='Amount to add')
    parser_credits_add.add_argument('-u', '--user-id', default='default_user', help='User ID')
    parser_credits_add.add_argument('-m', '--description', help='Transaction description')
    parser_credits_add.set_defaults(func=cmd_credits_add)

    # NEW: credits-history
    parser_credits_hist = subparsers.add_parser('credits-history', help='Show transaction history')
    parser_credits_hist.add_argument('-u', '--user-id', default='default_user', help='User ID')
    parser_credits_hist.add_argument('-l', '--limit', type=int, default=50, help='Limit results')
    parser_credits_hist.set_defaults(func=cmd_credits_history)

    # NEW: credits-reset
    parser_credits_reset = subparsers.add_parser('credits-reset', help='Reset credits')
    parser_credits_reset.add_argument('-u', '--user-id', default='default_user', help='User ID')
    parser_credits_reset.add_argument('-a', '--amount', type=int, help='Amount to reset to')
    parser_credits_reset.add_argument('-f', '--force', action='store_true', help='Force without confirmation')
    parser_credits_reset.set_defaults(func=cmd_credits_reset)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()