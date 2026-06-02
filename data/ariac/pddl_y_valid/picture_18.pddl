(define (problem picture_18)
    (:domain ariac)
    
    (:objects
        table pump_placement regulator_placement battery_placement buffer_placement - location
        red_battery green_battery blue_battery - part
    )
    
    (:init
        (robot_at table)
        (handempty)        
        (clear red_battery)
        (clear green_battery)
        (clear blue_battery)
        (part_at blue_battery table)
        (part_at green_battery table)
        (part_at red_battery table)
    )
    
    (:goal
        (and
            (part_at red_battery battery_placement)
        )
    )
)
